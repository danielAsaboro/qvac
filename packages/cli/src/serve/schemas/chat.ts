import { z } from 'zod'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { Tool } from '@qvac/sdk'
import {
  chatMessage,
  responseFormat,
  toolDef,
  openaiToolsToSdk,
  extractGenerationParams,
  extractResponseFormat,
  type GenerationParams,
  type ResponseFormat,
  type MessageContentPart
} from './common.js'

export const chatCompletionsBody = z.object({
  model: z.string().min(1),
  messages: z.array(chatMessage),
  stream: z.boolean().optional(),
  tools: z.array(toolDef).optional(),
  response_format: responseFormat.optional(),
  temperature: z.number().optional(),
  top_p: z.number().optional(),
  max_tokens: z.number().int().optional()
}).passthrough()

export const CHAT_UNSUPPORTED_PARAMS = [
  'logit_bias',
  'n',
  'user',
  'seed',
  'logprobs',
  'top_logprobs',
  'frequency_penalty',
  'presence_penalty',
  'stop'
] as const

interface OpenAIMessage {
  role: string
  content: string | null | undefined | MessageContentPart[]
  tool_calls?: Array<{
    id: string
    type: string
    function: { name: string; arguments: string }
  }>
  tool_call_id?: string
}

interface ChatHistoryItem {
  role: string
  content: string
  attachments?: Array<{ path: string }>
}

export function openaiMessagesToHistory (messages: OpenAIMessage[]): ChatHistoryItem[] {
  return messages.map((msg) => {
    if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
      return { role: 'assistant', content: synthesizeToolCallContent(msg.tool_calls) }
    }
    if (Array.isArray(msg.content)) {
      return multimodalContentToHistory(msg.role, msg.content)
    }
    return {
      role: msg.role,
      content: typeof msg.content === 'string' ? msg.content : (msg.content ?? '').toString()
    }
  })
}

// The inference image loader (stb_image, via llama.cpp) decodes PNG and JPEG. Other formats
// (e.g. webp) would fail to load and abort the completion mid-stream, so we only materialize
// supported types — keyed by media type → file extension.
const SUPPORTED_IMAGE_TYPES: Record<string, string> = {
  'image/jpeg': 'jpg',
  'image/png': 'png'
}

// OpenAI multimodal content is an array of parts. Concatenate the text and decode any
// `image_url` (base64 data URL) into a temp file referenced via SDK `attachments`, so
// multimodal models receive the image. Non-data URLs and unsupported image formats are
// skipped (the turn degrades to text-only rather than crashing the model).
function multimodalContentToHistory (role: string, parts: MessageContentPart[]): ChatHistoryItem {
  let content = ''
  const attachments: Array<{ path: string }> = []
  for (const part of parts) {
    if (part.type === 'text') {
      content += part.text
    } else if (part.type === 'image_url') {
      const url = typeof part.image_url === 'string' ? part.image_url : part.image_url.url
      const path = imageUrlToAttachmentPath(url)
      if (path !== undefined) attachments.push({ path })
    }
  }
  return attachments.length > 0 ? { role, content, attachments } : { role, content }
}

function imageUrlToAttachmentPath (url: string): string | undefined {
  const match = /^data:(image\/[a-z0-9.+-]+);base64,(.*)$/is.exec(url)
  if (match === null) return undefined
  const ext = SUPPORTED_IMAGE_TYPES[match[1].toLowerCase()]
  if (ext === undefined) return undefined
  const dir = mkdtempSync(join(tmpdir(), 'qvac-image-'))
  const file = join(dir, `image.${ext}`)
  writeFileSync(file, Buffer.from(match[2], 'base64'))
  return file
}

function synthesizeToolCallContent (toolCalls: NonNullable<OpenAIMessage['tool_calls']>): string {
  return toolCalls.map((tc) => {
    let args: Record<string, unknown>
    try {
      args = JSON.parse(tc.function.arguments) as Record<string, unknown>
    } catch {
      args = {}
    }
    const callObj = { name: tc.function.name, arguments: args }
    return `<tool_call>\n${JSON.stringify(callObj)}\n</tool_call>`
  }).join('\n')
}

export type ChatCompletionsBody = z.infer<typeof chatCompletionsBody>

export interface SdkChatArgs {
  history: ChatHistoryItem[]
  tools: Tool[] | undefined
  generationParams: GenerationParams | undefined
  responseFormat: ResponseFormat | undefined
  stream: boolean
}

export function toSdkChatArgs (body: ChatCompletionsBody): SdkChatArgs {
  const responseFmt = extractResponseFormat(body as Record<string, unknown>)
  return {
    history: openaiMessagesToHistory(body.messages as OpenAIMessage[]),
    tools: openaiToolsToSdk(body.tools as Parameters<typeof openaiToolsToSdk>[0]),
    generationParams: extractGenerationParams(body as Record<string, unknown>, 'max_completion_tokens'),
    responseFormat: responseFmt,
    stream: Boolean(body.stream)
  }
}
