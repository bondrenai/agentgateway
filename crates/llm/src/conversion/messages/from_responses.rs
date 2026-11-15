//! OpenAI Responses requests served by an Anthropic Messages upstream.
//!
//! Only buffered replies are translated. There is no conversion of the Messages event stream into
//! Responses events, so a streaming request is refused before it reaches the provider.

use std::collections::HashMap;

use agent_core::strng;
use bytes::Bytes;

use crate::conversion::bedrock::ReasoningCarrier;
use crate::conversion::completions::parse_data_url;
use crate::types::ResponseType;
use crate::types::messages::typed as messages;
use crate::types::responses::typed as responses;
use crate::{AIError, json, logged_response_parsing, types};

/// translate an OpenAI responses request to an anthropic messages request
pub fn translate(
	req: &types::responses::Request,
	prompt_caching: Option<&crate::PromptCachingConfig>,
	catalog: crate::model_catalog::Catalog<'_>,
) -> Result<Vec<u8>, AIError> {
	if req.stream == Some(true) {
		return Err(AIError::UnsupportedConversion(strng::literal!(
			"streaming responses to messages"
		)));
	}
	let typed = json::convert::<_, responses::CreateResponse>(req)
		.map_err(|err| AIError::RequestParsing(crate::InputFormat::Responses, err))?;
	let explicit_budget = req
		.vendor_extensions
		.as_ref()
		.and_then(|v| v.thinking_budget_tokens);
	let model_id = typed.model.clone().unwrap_or_default();
	let mut xlated = translate_internal(typed, explicit_budget, model_id, catalog)?;
	if let Some(caching) = prompt_caching {
		super::from_completions::apply_prompt_caching(&mut xlated, caching);
	}
	serde_json::to_vec(&xlated).map_err(AIError::RequestMarshal)
}

fn cache_control(
	breakpoint: &Option<responses::PromptCacheBreakpointConfig>,
) -> Option<messages::CacheControlEphemeral> {
	breakpoint
		.as_ref()
		.map(|_| messages::CacheControlEphemeral::Ephemeral { ttl: None })
}

fn text_block(
	text: String,
	cache_control: Option<messages::CacheControlEphemeral>,
) -> Option<messages::ContentBlock> {
	// The provider rejects empty text blocks.
	(!text.trim().is_empty()).then(|| {
		messages::ContentBlock::Text(messages::ContentTextBlock {
			text,
			citations: None,
			cache_control,
		})
	})
}

/// Base64 `data:` URIs become inline sources; anything else is passed as a URL for the provider
/// to fetch.
fn media_source(url: &str) -> serde_json::Value {
	match parse_data_url(url) {
		Some((media_type, data)) => serde_json::json!({
			"type": "base64",
			"media_type": media_type,
			"data": data
		}),
		None => serde_json::json!({
			"type": "url",
			"url": url
		}),
	}
}

/// Content parts of a user message or a tool output.
fn input_parts_to_messages(
	parts: Vec<responses::InputContent>,
) -> Result<Vec<messages::ContentBlock>, AIError> {
	let mut out = Vec::new();
	for part in parts {
		match part {
			responses::InputContent::InputText(text) => {
				out.extend(text_block(
					text.text,
					cache_control(&text.prompt_cache_breakpoint),
				));
			},
			responses::InputContent::InputImage(image) => {
				let Some(url) = image.image_url else {
					return Err(AIError::UnsupportedConversion(strng::literal!(
						"anthropic path cannot resolve an OpenAI image file_id; send image_url as a URL or an inline data: URI"
					)));
				};
				out.push(messages::ContentBlock::Image(messages::ContentImageBlock {
					source: media_source(&url),
					cache_control: cache_control(&image.prompt_cache_breakpoint),
				}));
			},
			responses::InputContent::InputFile(file) => {
				let source = match (
					file.file_data.as_deref().and_then(parse_data_url),
					file.file_url.as_deref(),
				) {
					(Some((media_type, data)), _) => serde_json::json!({
						"type": "base64",
						"media_type": media_type,
						"data": data
					}),
					(None, Some(url)) => media_source(url),
					(None, None) => {
						return Err(AIError::UnsupportedConversion(strng::literal!(
							"input_file requires file_data as an inline base64 data: URI, or file_url"
						)));
					},
				};
				out.push(messages::ContentBlock::Document(
					messages::ContentDocumentBlock {
						source,
						cache_control: cache_control(&file.prompt_cache_breakpoint),
						citations: None,
						context: None,
						title: file.filename,
					},
				));
			},
		}
	}
	Ok(out)
}

fn easy_content_to_messages(
	content: responses::EasyInputContent,
) -> Result<Vec<messages::ContentBlock>, AIError> {
	match content {
		responses::EasyInputContent::Text(text) => Ok(text_block(text, None).into_iter().collect()),
		responses::EasyInputContent::ContentList(parts) => input_parts_to_messages(parts),
	}
}

/// System and developer messages only carry text.
fn system_parts_to_messages(
	parts: Vec<responses::InputContent>,
) -> Result<Vec<messages::SystemContentBlock>, AIError> {
	let mut out = Vec::new();
	for part in parts {
		let responses::InputContent::InputText(text) = part else {
			return Err(AIError::UnsupportedConversion(strng::literal!(
				"anthropic system prompts only support text content"
			)));
		};
		if !text.text.trim().is_empty() {
			out.push(messages::SystemContentBlock::Text {
				text: text.text,
				cache_control: cache_control(&text.prompt_cache_breakpoint),
			});
		}
	}
	Ok(out)
}

fn tool_output_to_messages(
	output: responses::FunctionCallOutput,
) -> Result<messages::ToolResultContent, AIError> {
	let parts = match output {
		responses::FunctionCallOutput::Text(text) => {
			return Ok(messages::ToolResultContent::Text(text));
		},
		responses::FunctionCallOutput::Content(parts) => input_parts_to_messages(parts)?,
	};
	let parts = parts
		.into_iter()
		.filter_map(|block| match block {
			messages::ContentBlock::Text(text) => Some(messages::ToolResultContentPart::Text {
				text: text.text,
				citations: None,
				cache_control: text.cache_control,
			}),
			messages::ContentBlock::Image(image) => Some(messages::ToolResultContentPart::Image {
				source: image.source,
				cache_control: image.cache_control,
			}),
			messages::ContentBlock::Document(document) => {
				Some(messages::ToolResultContentPart::Document {
					source: document.source,
					cache_control: document.cache_control,
					citations: None,
					context: None,
					title: document.title,
				})
			},
			_ => None,
		})
		.collect();
	Ok(messages::ToolResultContent::Array(parts))
}

/// Responses lists a turn's reasoning, text, and tool calls as separate items; Messages carries
/// them as the blocks of one message, so content for a repeated role extends the previous message.
fn push_blocks(
	out: &mut Vec<messages::Message>,
	role: messages::Role,
	content: Vec<messages::ContentBlock>,
) {
	if content.is_empty() {
		return;
	}
	match out.last_mut() {
		Some(last) if last.role == role => last.content.extend(content),
		_ => out.push(messages::Message { role, content }),
	}
}

fn translate_internal(
	req: responses::CreateResponse,
	explicit_budget: Option<u64>,
	model_id: String,
	catalog: crate::model_catalog::Catalog<'_>,
) -> Result<messages::Request, AIError> {
	use responses::{
		InputItem, InputParam, InputRole, Item, MessageItem, OutputMessageContent, Role,
	};

	let max_tokens = req.max_output_tokens.unwrap_or(4096) as usize;

	let mut system_blocks = Vec::new();
	if let Some(instructions) = req.instructions
		&& !instructions.trim().is_empty()
	{
		system_blocks.push(messages::SystemContentBlock::Text {
			text: instructions,
			cache_control: None,
		});
	}

	let mut out: Vec<messages::Message> = Vec::new();
	let items = match req.input {
		InputParam::Text(text) => {
			push_blocks(
				&mut out,
				messages::Role::User,
				text_block(text, None).into_iter().collect(),
			);
			Vec::new()
		},
		InputParam::Items(items) => items,
	};
	for item in items {
		match item {
			InputItem::EasyMessage(msg) => match msg.role {
				Role::System | Role::Developer => match msg.content {
					responses::EasyInputContent::Text(text) => {
						if !text.trim().is_empty() {
							system_blocks.push(messages::SystemContentBlock::Text {
								text,
								cache_control: None,
							});
						}
					},
					responses::EasyInputContent::ContentList(parts) => {
						system_blocks.extend(system_parts_to_messages(parts)?);
					},
				},
				Role::User => push_blocks(
					&mut out,
					messages::Role::User,
					easy_content_to_messages(msg.content)?,
				),
				Role::Assistant => push_blocks(
					&mut out,
					messages::Role::Assistant,
					easy_content_to_messages(msg.content)?,
				),
			},
			InputItem::Item(Item::Message(MessageItem::Input(msg))) => match msg.role {
				InputRole::System | InputRole::Developer => {
					system_blocks.extend(system_parts_to_messages(msg.content)?);
				},
				InputRole::User => push_blocks(
					&mut out,
					messages::Role::User,
					input_parts_to_messages(msg.content)?,
				),
			},
			InputItem::Item(Item::Message(MessageItem::Output(msg))) => {
				let content = msg
					.content
					.into_iter()
					.filter_map(|part| match part {
						OutputMessageContent::OutputText(text) => text_block(text.text, None),
						OutputMessageContent::Refusal(refusal) => text_block(refusal.refusal, None),
					})
					.collect();
				push_blocks(&mut out, messages::Role::Assistant, content);
			},
			InputItem::Item(Item::Reasoning(reasoning)) => {
				// Thinking is replayed from the carrier the response translation put in
				// encrypted_content. The provider rejects thinking without its signature, so plain
				// reasoning text and another provider's encrypted reasoning are dropped.
				let content = reasoning
					.encrypted_content
					.as_deref()
					.and_then(ReasoningCarrier::decode)
					.unwrap_or_default()
					.into_iter()
					.map(|block| match block {
						ReasoningCarrier::Thinking {
							thinking,
							signature,
						} => messages::ContentBlock::Thinking {
							thinking,
							signature,
						},
						ReasoningCarrier::RedactedThinking { data } => {
							messages::ContentBlock::RedactedThinking { data }
						},
					})
					.collect();
				push_blocks(&mut out, messages::Role::Assistant, content);
			},
			InputItem::Item(Item::FunctionCall(call)) => push_blocks(
				&mut out,
				messages::Role::Assistant,
				vec![messages::ContentBlock::ToolUse {
					input: crate::conversion::tool_arguments_to_input(&call.arguments),
					id: call.call_id,
					name: call.name,
					cache_control: None,
				}],
			),
			InputItem::Item(Item::FunctionCallOutput(output)) => {
				let Some(call_id) = output.call_id.filter(|call_id| !call_id.is_empty()) else {
					continue;
				};
				push_blocks(
					&mut out,
					messages::Role::User,
					vec![messages::ContentBlock::ToolResult {
						tool_use_id: call_id,
						content: tool_output_to_messages(output.output)?,
						cache_control: None,
						is_error: None,
					}],
				);
			},
			_ => {
				tracing::debug!("Skipping unsupported Responses input item for Messages translation");
			},
		}
	}

	let system = if system_blocks.is_empty() {
		None
	} else if system_blocks
		.iter()
		.any(|messages::SystemContentBlock::Text { cache_control, .. }| cache_control.is_some())
	{
		Some(messages::SystemPrompt::Blocks(system_blocks))
	} else {
		Some(messages::SystemPrompt::Text(
			system_blocks
				.into_iter()
				.map(|messages::SystemContentBlock::Text { text, .. }| text)
				.collect::<Vec<_>>()
				.join("\n"),
		))
	};

	let tools: Vec<_> = req
		.tools
		.iter()
		.flatten()
		.filter_map(|tool| match tool {
			responses::Tool::Function(function) => Some(messages::Tool::Custom(messages::CustomTool {
				name: function.name.clone(),
				description: function.description.clone(),
				input_schema: function
					.parameters
					.clone()
					.unwrap_or_else(|| serde_json::json!({"type": "object"})),
				strict: function.strict,
				cache_control: None,
			})),
			_ => {
				tracing::warn!("Unsupported tool type in Responses API: {:?}", tool);
				None
			},
		})
		.collect();

	let disable_parallel_tool_use = req.parallel_tool_calls.map(|p| !p);
	// The provider only accepts a tool choice alongside tools.
	let tool_choice = if tools.is_empty() {
		None
	} else {
		use responses::{ToolChoiceFunction, ToolChoiceOptions, ToolChoiceParam};
		match req.tool_choice {
			Some(ToolChoiceParam::Function(ToolChoiceFunction { name })) => {
				Some(messages::ToolChoice::Tool {
					name,
					disable_parallel_tool_use,
				})
			},
			Some(ToolChoiceParam::Option(ToolChoiceOptions::Auto)) => Some(messages::ToolChoice::Auto {
				disable_parallel_tool_use,
			}),
			Some(ToolChoiceParam::Option(ToolChoiceOptions::Required)) => {
				Some(messages::ToolChoice::Any {
					disable_parallel_tool_use,
				})
			},
			Some(ToolChoiceParam::Option(ToolChoiceOptions::None)) => Some(messages::ToolChoice::None {}),
			Some(other) => {
				tracing::warn!("Unsupported tool choice for Messages: {:?}", other);
				None
			},
			None if disable_parallel_tool_use.is_some() => Some(messages::ToolChoice::Auto {
				disable_parallel_tool_use,
			}),
			None => None,
		}
	};

	let metadata = req.safety_identifier.map(|user| messages::Metadata {
		fields: HashMap::from([("user_id".to_string(), user)]),
	});

	let capabilities = crate::model_catalog::anthropic_thinking_capabilities(&model_id, catalog);
	let effort = req
		.reasoning
		.as_ref()
		.and_then(|reasoning| reasoning.effort.as_ref())
		.and_then(crate::types::anthropic_effort_for_reasoning_effort);
	let (thinking, effort) = if let Some(budget_tokens) = explicit_budget
		&& capabilities.legacy
	{
		(
			super::cap_thinking_budget_to_max_tokens(budget_tokens, max_tokens)
				.map(|budget_tokens| messages::ThinkingInput::Enabled { budget_tokens }),
			None,
		)
	} else if (explicit_budget.is_some() || effort.is_some()) && capabilities.adaptive {
		(
			Some(messages::ThinkingInput::Adaptive {}),
			Some(effort.unwrap_or(messages::ThinkingEffort::High)),
		)
	} else {
		let budget_tokens =
			explicit_budget.or_else(|| effort.map(crate::types::thinking_budget_for_anthropic_effort));
		(
			budget_tokens
				.and_then(|budget_tokens| {
					super::cap_thinking_budget_to_max_tokens(budget_tokens, max_tokens)
				})
				.map(|budget_tokens| messages::ThinkingInput::Enabled { budget_tokens }),
			None,
		)
	};

	let format = req.text.and_then(|text| match text.format {
		responses::TextResponseFormatConfiguration::JsonSchema(json_schema) => {
			Some(messages::OutputFormat::JsonSchema {
				schema: json_schema.schema,
			})
		},
		responses::TextResponseFormatConfiguration::JsonObject => {
			Some(messages::OutputFormat::JsonSchema {
				schema: serde_json::json!({
					"type": "object",
					"additionalProperties": true
				}),
			})
		},
		responses::TextResponseFormatConfiguration::Text => None,
	});
	let output_config = if format.is_some() || effort.is_some() {
		Some(messages::OutputConfig { effort, format })
	} else {
		None
	};

	// Request fields with no Messages equivalent (service_tier, store, include, truncation, and
	// the like) are dropped rather than rejected.
	Ok(messages::Request {
		messages: out,
		system,
		model: model_id,
		max_tokens,
		stop_sequences: Vec::new(),
		stream: false,
		temperature: req.temperature,
		top_p: req.top_p,
		top_k: None,
		tools: if tools.is_empty() { None } else { Some(tools) },
		tool_choice,
		metadata,
		thinking,
		output_config,
		cache_control: None,
	})
}

pub fn translate_response(bytes: &Bytes) -> Result<Box<dyn ResponseType>, AIError> {
	let resp = serde_json::from_slice::<messages::MessagesResponse>(bytes)
		.map_err(logged_response_parsing(bytes))?;
	let usage = resp.usage.accounting();
	let reasoning_tokens = thinking_tokens(bytes);
	let provider_usage = Some(super::super::ProviderUsage {
		input_tokens: usage.input_tokens,
		total_tokens: usage.input_tokens.saturating_add(usage.output_tokens),
		reasoning_tokens,
		cache_creation_1h_input_tokens: usage.cache_creation_1h_input_tokens,
		usage_lines: crate::types::messages::usage_lines(&resp.usage.iterations),
		..Default::default()
	});
	let typed = translate_response_internal(resp, reasoning_tokens);
	let passthrough =
		json::convert::<_, types::responses::Response>(&typed).map_err(AIError::ResponseParsing)?;
	Ok(Box::new(super::super::ResponseWithProviderUsage {
		response: passthrough,
		provider_usage,
	}))
}

/// The provider reports thinking tokens in `usage.output_tokens_details`, which the typed usage
/// does not model.
fn thinking_tokens(bytes: &[u8]) -> Option<u64> {
	#[derive(serde::Deserialize)]
	struct Response {
		usage: Usage,
	}
	#[derive(serde::Deserialize)]
	struct Usage {
		output_tokens_details: Option<OutputTokensDetails>,
	}
	#[derive(serde::Deserialize)]
	struct OutputTokensDetails {
		thinking_tokens: Option<u64>,
	}
	serde_json::from_slice::<Response>(bytes)
		.ok()?
		.usage
		.output_tokens_details?
		.thinking_tokens
}

fn translate_response_internal(
	resp: messages::MessagesResponse,
	reasoning_tokens: Option<u64>,
) -> responses::Response {
	let accounting = resp.usage.accounting();
	let (status, incomplete_reason) = match resp.stop_reason {
		Some(messages::StopReason::MaxTokens | messages::StopReason::ModelContextWindowExceeded) => {
			(responses::Status::Incomplete, Some("max_output_tokens"))
		},
		Some(messages::StopReason::Refusal) => (responses::Status::Incomplete, Some("content_filter")),
		_ => (responses::Status::Completed, None),
	};
	let item_status = if incomplete_reason.is_some() {
		responses::OutputStatus::Incomplete
	} else {
		responses::OutputStatus::Completed
	};

	// Items keep the order of the provider's content blocks, so replaying them rebuilds the same
	// assistant message. Adjacent text blocks share one message item.
	let id = resp.id.strip_prefix("msg_").unwrap_or(&resp.id);
	let mut output: Vec<responses::OutputItem> = Vec::new();
	let mut text_parts: Vec<responses::OutputMessageContent> = Vec::new();
	let flush_text = |output: &mut Vec<responses::OutputItem>,
	                  text_parts: &mut Vec<responses::OutputMessageContent>| {
		if !text_parts.is_empty() {
			output.push(responses::OutputItem::Message(responses::OutputMessage {
				id: format!("msg_{id}_{}", output.len()),
				role: responses::AssistantRole::Assistant,
				phase: None,
				content: std::mem::take(text_parts),
				status: item_status,
			}));
		}
	};
	for block in resp.content {
		let item = match block {
			messages::ContentBlock::Text(text) => {
				text_parts.push(responses::OutputMessageContent::OutputText(
					responses::OutputTextContent {
						annotations: vec![],
						logprobs: None,
						text: text.text,
					},
				));
				continue;
			},
			// Signed thinking is returned in encrypted_content so a stateless client can replay it.
			messages::ContentBlock::Thinking {
				thinking,
				signature,
			} => {
				if thinking.is_empty() && signature.is_empty() {
					continue;
				}
				responses::OutputItem::Reasoning(responses::ReasoningItem {
					id: Some(format!("rs_{id}_{}", output.len())),
					summary: vec![],
					content: (!thinking.is_empty()).then(|| {
						vec![responses::ReasoningItemContent::ReasoningText(
							responses::ReasoningTextContent {
								text: thinking.clone(),
							},
						)]
					}),
					encrypted_content: (!signature.is_empty()).then(|| {
						ReasoningCarrier::Thinking {
							thinking,
							signature,
						}
						.encode()
					}),
					status: Some(item_status),
				})
			},
			messages::ContentBlock::RedactedThinking { data } => {
				responses::OutputItem::Reasoning(responses::ReasoningItem {
					id: Some(format!("rs_{id}_{}", output.len())),
					summary: vec![],
					content: None,
					encrypted_content: Some(ReasoningCarrier::RedactedThinking { data }.encode()),
					status: Some(item_status),
				})
			},
			messages::ContentBlock::ToolUse {
				id: call_id,
				name,
				input,
				..
			} => responses::OutputItem::FunctionCall(responses::FunctionToolCall {
				arguments: serde_json::to_string(&input).unwrap_or_default(),
				call_id: call_id.clone(),
				namespace: None,
				name,
				caller: None,
				id: Some(call_id),
				status: Some(item_status),
				r#async: None,
			}),
			// Nothing else is requested by the request translation.
			_ => continue,
		};
		flush_text(&mut output, &mut text_parts);
		output.push(item);
	}
	flush_text(&mut output, &mut text_parts);

	// Responses counts cached tokens as part of the input; Messages reports them separately.
	let input_tokens = accounting
		.input_tokens
		.saturating_add(accounting.cache_read_input_tokens.unwrap_or_default())
		.saturating_add(accounting.cache_creation_input_tokens.unwrap_or_default());
	let usage = responses::ResponseUsage {
		input_tokens: input_tokens as u32,
		output_tokens: accounting.output_tokens as u32,
		total_tokens: input_tokens.saturating_add(accounting.output_tokens) as u32,
		input_tokens_details: responses::InputTokenDetails {
			cached_tokens: accounting.cache_read_input_tokens.unwrap_or_default() as u32,
			cache_write_tokens: accounting
				.cache_creation_input_tokens
				.map(|tokens| tokens as i64),
			cache_write_tokens_details: None,
		},
		output_tokens_details: responses::OutputTokenDetails {
			reasoning_tokens: reasoning_tokens.unwrap_or_default() as u32,
		},
	};

	let builder = types::responses::ResponseBuilder::new(format!("resp_{id}"), resp.model);
	let mut response = builder.response(
		status,
		Some(usage),
		None,
		incomplete_reason.map(|reason| responses::IncompleteDetails {
			reason: reason.to_string(),
		}),
	);
	response.output = output;
	response.service_tier = match resp.usage.service_tier.as_deref() {
		Some("standard") => Some(responses::ServiceTierResponses::Default),
		Some("priority") => Some(responses::ServiceTierResponses::Priority),
		_ => None,
	};
	response
}

pub fn translate_error(bytes: &Bytes) -> Result<Bytes, AIError> {
	super::from_completions::translate_error(bytes)
}

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;

	fn request(body: serde_json::Value) -> types::responses::Request {
		serde_json::from_value(body).unwrap()
	}

	fn translated(req: &types::responses::Request) -> serde_json::Value {
		serde_json::from_slice(&translate(req, None, None).unwrap()).unwrap()
	}

	#[test]
	fn streaming_is_unsupported() {
		let req = request(json!({"model": "claude-opus-4-6", "input": "Hello", "stream": true}));
		let err = translate(&req, None, None).unwrap_err();
		assert!(matches!(err, AIError::UnsupportedConversion(_)), "{err}");
	}

	#[test]
	fn prompt_caching_policy_marks_system_history_and_tools() {
		let req = request(json!({
			"model": "claude-opus-4-6",
			"instructions": "You are helpful.",
			"input": [
				{"role": "user", "content": "Hello"},
				{"role": "assistant", "content": "Hi"},
				{"role": "user", "content": "How are you?"}
			],
			"tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]
		}));
		let caching = crate::PromptCachingConfig {
			cache_tools: true,
			min_tokens: None,
			..Default::default()
		};
		let out: serde_json::Value =
			serde_json::from_slice(&translate(&req, Some(&caching), None).unwrap()).unwrap();
		let mark = json!({"type": "ephemeral"});
		assert_eq!(out["system"][0]["cache_control"], mark);
		assert_eq!(out["messages"][1]["content"][0]["cache_control"], mark);
		assert!(
			out["messages"][2]["content"][0]
				.get("cache_control")
				.is_none()
		);
		assert_eq!(out["tools"][0]["cache_control"], mark);
	}

	// A client that sends a reply's output items back, as stateless Responses clients do, must
	// produce the assistant message the provider returned: it verifies thinking signatures.
	#[test]
	fn replayed_output_rebuilds_the_assistant_message() {
		let content = json!([
			{"type": "redacted_thinking", "data": "cmVkYWN0ZWQ="},
			{"type": "thinking", "thinking": "Let me check.", "signature": "SIG-xyz=="},
			{"type": "text", "text": "Checking."},
			{"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"location": "Columbus, OH"}}
		]);
		let reply = json!({
			"id": "msg_01",
			"type": "message",
			"role": "assistant",
			"model": "claude-opus-4-6",
			"content": content,
			"stop_reason": "tool_use",
			"stop_sequence": null,
			"usage": {"input_tokens": 1, "output_tokens": 1}
		});
		let reply = translate_response(&Bytes::from(serde_json::to_vec(&reply).unwrap()))
			.unwrap()
			.serialize()
			.unwrap();
		let reply: serde_json::Value = serde_json::from_slice(&reply).unwrap();

		let mut input = vec![json!({"role": "user", "content": "Weather?"})];
		input.extend(reply["output"].as_array().unwrap().iter().cloned());
		input.push(json!({"type": "function_call_output", "call_id": "toolu_01", "output": "sunny"}));
		let out = translated(&request(
			json!({"model": "claude-opus-4-6", "input": input}),
		));
		assert_eq!(
			out["messages"][1],
			json!({"role": "assistant", "content": content})
		);
		assert_eq!(
			out["messages"][2]["content"],
			json!([{"type": "tool_result", "tool_use_id": "toolu_01", "content": "sunny"}])
		);
	}

	#[test]
	fn refusal_is_reported_as_content_filter() {
		let reply = json!({
			"id": "msg_01",
			"type": "message",
			"role": "assistant",
			"model": "claude-opus-4-6",
			"content": [],
			"stop_reason": "refusal",
			"stop_sequence": null,
			"usage": {"input_tokens": 1, "output_tokens": 0}
		});
		let reply = translate_response(&Bytes::from(serde_json::to_vec(&reply).unwrap()))
			.unwrap()
			.serialize()
			.unwrap();
		let reply: serde_json::Value = serde_json::from_slice(&reply).unwrap();
		assert_eq!(reply["status"], "incomplete");
		assert_eq!(reply["incomplete_details"]["reason"], "content_filter");
	}
}
