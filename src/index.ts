/**
 * 🚀 Universal LLM
 * Edge-ready, multi-provider LLM orchestrator with structured output.
 *
 * Supports: OpenAI, Anthropic, Grok (xAI), Ollama
 * Features: Streaming, Caching, Retries, Zod Structured Output, Middleware
 */

import { z } from 'zod';

// ──────────────────────────────────────────────
// 1. TYPES
// ──────────────────────────────────────────────

export type ModelProvider = 'openai' | 'anthropic' | 'grok' | 'ollama';

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LLMOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
}

export interface LLMConfig {
  provider: ModelProvider;
  apiKey?: string;
  baseUrl?: string;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  cache?: boolean;
  cacheTTL?: number;
  onRequest?: (payload: Record<string, unknown>) => Record<string, unknown> | void;
  onResponse?: (response: string) => string | void;
  onError?: (error: Error) => void;
}

export interface StructuredOutputOptions<T extends z.ZodType> extends LLMOptions {
  schema: T;
  prompt: string;
  system?: string;
}

export interface CacheEntry {
  value: string;
  expiresAt: number;
}

export interface LLMResult<T> {
  data: T;
  metadata: {
    durationMs: number;
    provider: ModelProvider;
    model: string;
    cached: boolean;
  };
}

// ──────────────────────────────────────────────
// 2. ZOD → JSON SCHEMA CONVERTER
//    (No reliance on Zod internals)
// ──────────────────────────────────────────────

function zodToJsonDescription(schema: z.ZodType): string {
  // 1. If it's an object, walk its shape (Most common case for structured output)
  if ('shape' in schema && typeof schema.shape === 'object') {
    const shape = schema.shape as Record<string, z.ZodType>;
    const entries = Object.entries(shape).map(([key, val]) => {
      const typeName = getZodTypeName(val);
      return `"${key}": ${typeName}`;
    });
    return `{ ${entries.join(', ')} }`;
  }

  // 2. Fallback: Try to parse undefined to see required fields
  try {
    const result = schema.safeParse(undefined);
    if (!result.success) {
      return `{ ${result.error.issues.map(i => `"${i.path.join('.') || 'value'}": ${i.message}`).join(', ')} }`;
    }
  } catch {
    // Silent fallback
  }

  return 'JSON object';
}

function getZodTypeName(schema: z.ZodType): string {
  const def = (schema as any)._def;
  const typeName = def?.typeName;
  
  switch (typeName) {
    case 'ZodString': return 'string';
    case 'ZodNumber': return 'number';
    case 'ZodBoolean': return 'boolean';
    case 'ZodEnum': return (def.values as string[]).map(v => `'${v}'`).join(' | ');
    case 'ZodArray': return `${getZodTypeName(def.type)}[]`;
    case 'ZodOptional': return `${getZodTypeName(def.innerType)} (optional)`;
    case 'ZodObject': return 'object';
    default: return 'value';
  }
}

// ──────────────────────────────────────────────
// 3. IN-MEMORY CACHE
// ──────────────────────────────────────────────

class LLMCache {
  private store = new Map<string, CacheEntry>();

  private hash(messages: LLMMessage[], options?: LLMOptions): string {
    const raw = JSON.stringify({ messages, options });
    // Simple djb2 hash — no crypto dependency needed
    let hash = 5381;
    for (let i = 0; i < raw.length; i++) {
      hash = ((hash << 5) + hash + raw.charCodeAt(i)) & 0xffffffff;
    }
    return hash.toString(36);
  }

  get(messages: LLMMessage[], options?: LLMOptions): string | null {
    const key = this.hash(messages, options);
    const entry = this.store.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.store.delete(key);
      return null;
    }
    return entry.value;
  }

  set(messages: LLMMessage[], options: LLMOptions | undefined, value: string, ttl: number): void {
    const key = this.hash(messages, options);
    this.store.set(key, { value, expiresAt: Date.now() + ttl });
  }

  clear(): void {
    this.store.clear();
  }
}

// ──────────────────────────────────────────────
// 4. RETRY WITH EXPONENTIAL BACKOFF
// ──────────────────────────────────────────────

async function withRetry<T>(
  fn: () => Promise<T>,
  retries: number,
  delay: number,
  onError?: (error: Error) => void
): Promise<T> {
  let lastError: Error | undefined;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (onError) onError(lastError);
      if (attempt < retries) {
        const backoff = delay * Math.pow(2, attempt);
        await new Promise(r => setTimeout(r, backoff));
      }
    }
  }
  throw lastError;
}

// ──────────────────────────────────────────────
// 5. FETCH WITH TIMEOUT
// ──────────────────────────────────────────────

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number
): Promise<Response> {
  if (timeoutMs <= 0) return fetch(url, init);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    if ((err as Error).name === 'AbortError') {
      throw new Error(`[UniversalLLM] Request timed out after ${timeoutMs}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

// ──────────────────────────────────────────────
// 6. CORE CLIENT
// ──────────────────────────────────────────────

export class UniversalLLM {
  private apiKey: string;
  private provider: ModelProvider;
  private baseUrl?: string;
  private timeout: number;
  private retries: number;
  private retryDelay: number;
  private cacheEnabled: boolean;
  private cacheTTL: number;
  private cache: LLMCache;
  private onRequest?: (payload: Record<string, unknown>) => Record<string, unknown> | void;
  private onResponse?: (response: string) => string | void;
  private onError?: (error: Error) => void;

  constructor(config: LLMConfig) {
    this.provider = config.provider;
    this.apiKey = config.apiKey || '';
    this.baseUrl = config.baseUrl;
    this.timeout = config.timeout ?? 30000;
    this.retries = config.retries ?? 2;
    this.retryDelay = config.retryDelay ?? 1000;
    this.cacheEnabled = config.cache ?? false;
    this.cacheTTL = config.cacheTTL ?? 300000; // 5 minutes default
    this.cache = new LLMCache();
    this.onRequest = config.onRequest;
    this.onResponse = config.onResponse;
    this.onError = config.onError;
  }

  /**
   * Simple chat completion — returns a string response with metadata.
   */
  async chat(messages: LLMMessage[], options?: LLMOptions): Promise<LLMResult<string>> {
    const start = Date.now();
    let cached = false;
    let result: string;

    // Check cache
    if (this.cacheEnabled) {
      const entry = this.cache.get(messages, options);
      if (entry) {
        result = entry;
        cached = true;
      } else {
        result = await withRetry(
          () => this.dispatch(messages, options),
          this.retries,
          this.retryDelay,
          this.onError
        );
      }
    } else {
      result = await withRetry(
        () => this.dispatch(messages, options),
        this.retries,
        this.retryDelay,
        this.onError
      );
    }

    // Apply response middleware
    const finalResult = this.onResponse ? (this.onResponse(result) ?? result) : result;

    // Store in cache
    if (this.cacheEnabled && !cached) {
      this.cache.set(messages, options, finalResult, this.cacheTTL);
    }

    return {
      data: finalResult,
      metadata: {
        durationMs: Date.now() - start,
        provider: this.provider,
        model: options?.model || 'default',
        cached
      }
    };
  }

  /**
   * Streaming chat completion — yields string chunks.
   */
  async *stream(messages: LLMMessage[], options?: LLMOptions): AsyncGenerator<string, void, unknown> {
    switch (this.provider) {
      case 'openai':
      case 'grok':
        yield* this.streamOpenAICompatible(messages, options);
        break;
      case 'anthropic':
        yield* this.streamAnthropic(messages, options);
        break;
      case 'ollama':
        yield* this.streamOllama(messages, options);
        break;
      default:
        throw new Error(`[UniversalLLM] Streaming not supported for provider: ${this.provider}`);
    }
  }

  /**
   * Structured output — returns a Zod-validated, fully-typed object with metadata.
   */
  async structured<T extends z.ZodType>(options: StructuredOutputOptions<T>): Promise<LLMResult<z.infer<T>>> {
    const start = Date.now();
    const schemaDesc = zodToJsonDescription(options.schema);

    const systemPrompt = options.system || 'You are a helpful assistant that outputs ONLY valid JSON.';
    const finalPrompt = `${options.prompt}\n\n### RESPONSE REQUIREMENT\nYou must respond with a valid JSON object. No conversation, no markdown, no code blocks.\n\n### EXPECTED STRUCTURE\n${schemaDesc}`;

    const response = await this.chat([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: finalPrompt }
    ], { ...options, temperature: 0 });

    try {
      // Strip any accidental markdown fences
      const cleanJson = response.data
        .replace(/^```(?:json)?\s*/i, '')
        .replace(/\s*```$/i, '')
        .trim();
      const parsed = JSON.parse(cleanJson);
      const validated = options.schema.parse(parsed);

      return {
        data: validated,
        metadata: {
          ...response.metadata,
          durationMs: Date.now() - start // Total duration including nested chat call
        }
      };
    } catch (e) {
      throw new Error(
        `[UniversalLLM] Structured output failed.\n` +
        `Validation: ${e instanceof Error ? e.message : String(e)}\n` +
        `Raw response: ${response.data.substring(0, 500)}`
      );
    }
  }

  /**
   * Clear the response cache.
   */
  clearCache(): void {
    this.cache.clear();
  }

  // ──────────────────────────────────────────────
  // INTERNAL: DISPATCH TO PROVIDER
  // ──────────────────────────────────────────────

  private async dispatch(messages: LLMMessage[], options?: LLMOptions): Promise<string> {
    switch (this.provider) {
      case 'openai':
        return this.callOpenAI(messages, options);
      case 'anthropic':
        return this.callAnthropic(messages, options);
      case 'grok':
        return this.callGrok(messages, options);
      case 'ollama':
        return this.callOllama(messages, options);
      default:
        throw new Error(`[UniversalLLM] Unknown provider: ${this.provider}`);
    }
  }

  // ──────────────────────────────────────────────
  // PROVIDER: OpenAI
  // ──────────────────────────────────────────────

  private async callOpenAI(messages: LLMMessage[], options?: LLMOptions): Promise<string> {
    const url = this.baseUrl || 'https://api.openai.com/v1/chat/completions';
    let payload: Record<string, unknown> = {
      model: options?.model || 'gpt-4o',
      messages,
      temperature: options?.temperature ?? 0.7,
      max_tokens: options?.maxTokens,
      top_p: options?.topP,
    };

    if (this.onRequest) {
      payload = this.onRequest(payload) ?? payload;
    }

    const response = await fetchWithTimeout(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify(payload)
    }, this.timeout);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(`[OpenAI] ${(error as any).error?.message || response.statusText} (${response.status})`);
    }

    const data = await response.json();
    return (data as any).choices[0].message.content;
  }

  // ──────────────────────────────────────────────
  // PROVIDER: Anthropic
  // ──────────────────────────────────────────────

  private async callAnthropic(messages: LLMMessage[], options?: LLMOptions): Promise<string> {
    const url = this.baseUrl || 'https://api.anthropic.com/v1/messages';
    let payload: Record<string, unknown> = {
      model: options?.model || 'claude-sonnet-4-20250514',
      messages: messages.filter(m => m.role !== 'system'),
      system: messages.find(m => m.role === 'system')?.content,
      max_tokens: options?.maxTokens || 1024,
      temperature: options?.temperature ?? 0.7,
      top_p: options?.topP,
    };

    if (this.onRequest) {
      payload = this.onRequest(payload) ?? payload;
    }

    const response = await fetchWithTimeout(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify(payload)
    }, this.timeout);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(`[Anthropic] ${(error as any).error?.message || response.statusText} (${response.status})`);
    }

    const data = await response.json();
    return (data as any).content[0].text;
  }

  // ──────────────────────────────────────────────
  // PROVIDER: Grok (xAI) — OpenAI-compatible API
  // ──────────────────────────────────────────────

  private async callGrok(messages: LLMMessage[], options?: LLMOptions): Promise<string> {
    const url = this.baseUrl || 'https://api.x.ai/v1/chat/completions';
    let payload: Record<string, unknown> = {
      model: options?.model || 'grok-3',
      messages,
      temperature: options?.temperature ?? 0.7,
      max_tokens: options?.maxTokens,
    };

    if (this.onRequest) {
      payload = this.onRequest(payload) ?? payload;
    }

    const response = await fetchWithTimeout(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify(payload)
    }, this.timeout);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(`[Grok] ${(error as any).error?.message || response.statusText} (${response.status})`);
    }

    const data = await response.json();
    return (data as any).choices[0].message.content;
  }

  // ──────────────────────────────────────────────
  // PROVIDER: Ollama (Local)
  // ──────────────────────────────────────────────

  private async callOllama(messages: LLMMessage[], options?: LLMOptions): Promise<string> {
    const url = this.baseUrl || 'http://localhost:11434/api/chat';
    let payload: Record<string, unknown> = {
      model: options?.model || 'llama3',
      messages,
      stream: false,
      options: {
        temperature: options?.temperature ?? 0.7,
        num_predict: options?.maxTokens,
      }
    };

    if (this.onRequest) {
      payload = this.onRequest(payload) ?? payload;
    }

    const response = await fetchWithTimeout(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }, this.timeout);

    if (!response.ok) {
      throw new Error(`[Ollama] ${response.statusText} (${response.status})`);
    }

    const data = await response.json();
    return (data as any).message.content;
  }

  // ──────────────────────────────────────────────
  // STREAMING: OpenAI-Compatible (OpenAI + Grok)
  // ──────────────────────────────────────────────

  private async *streamOpenAICompatible(messages: LLMMessage[], options?: LLMOptions): AsyncGenerator<string> {
    const isGrok = this.provider === 'grok';
    const url = this.baseUrl || (isGrok
      ? 'https://api.x.ai/v1/chat/completions'
      : 'https://api.openai.com/v1/chat/completions');

    const response = await fetchWithTimeout(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        model: options?.model || (isGrok ? 'grok-3' : 'gpt-4o'),
        messages,
        temperature: options?.temperature ?? 0.7,
        max_tokens: options?.maxTokens,
        stream: true,
      })
    }, this.timeout);

    if (!response.ok || !response.body) {
      throw new Error(`[${isGrok ? 'Grok' : 'OpenAI'}] Stream failed: ${response.statusText}`);
    }

    yield* this.parseSSE(response.body, (json) => {
      if (json === '[DONE]') return null;
      const parsed = JSON.parse(json);
      return parsed.choices?.[0]?.delta?.content || null;
    });
  }

  // ──────────────────────────────────────────────
  // STREAMING: Anthropic
  // ──────────────────────────────────────────────

  private async *streamAnthropic(messages: LLMMessage[], options?: LLMOptions): AsyncGenerator<string> {
    const url = this.baseUrl || 'https://api.anthropic.com/v1/messages';

    const response = await fetchWithTimeout(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: options?.model || 'claude-sonnet-4-20250514',
        messages: messages.filter(m => m.role !== 'system'),
        system: messages.find(m => m.role === 'system')?.content,
        max_tokens: options?.maxTokens || 1024,
        temperature: options?.temperature ?? 0.7,
        stream: true,
      })
    }, this.timeout);

    if (!response.ok || !response.body) {
      throw new Error(`[Anthropic] Stream failed: ${response.statusText}`);
    }

    yield* this.parseSSE(response.body, (json) => {
      const parsed = JSON.parse(json);
      if (parsed.type === 'content_block_delta') {
        return parsed.delta?.text || null;
      }
      return null;
    });
  }

  // ──────────────────────────────────────────────
  // STREAMING: Ollama
  // ──────────────────────────────────────────────

  private async *streamOllama(messages: LLMMessage[], options?: LLMOptions): AsyncGenerator<string> {
    const url = this.baseUrl || 'http://localhost:11434/api/chat';

    const response = await fetchWithTimeout(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: options?.model || 'llama3',
        messages,
        stream: true,
        options: {
          temperature: options?.temperature ?? 0.7,
          num_predict: options?.maxTokens,
        }
      })
    }, this.timeout);

    if (!response.ok || !response.body) {
      throw new Error(`[Ollama] Stream failed: ${response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        for (const line of text.split('\n')) {
          if (!line.trim()) continue;
          try {
            const parsed = JSON.parse(line);
            if (parsed.message?.content) {
              yield parsed.message.content;
            }
          } catch {
            // Skip malformed lines
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // ──────────────────────────────────────────────
  // SSE PARSER (Shared by OpenAI + Anthropic)
  // ──────────────────────────────────────────────

  private async *parseSSE(
    body: ReadableStream<Uint8Array>,
    extractor: (json: string) => string | null
  ): AsyncGenerator<string> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (!data || data === '[DONE]') continue;

          try {
            const chunk = extractor(data);
            if (chunk) yield chunk;
          } catch {
            // Skip malformed SSE data
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}
