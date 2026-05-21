# 🚀 Universal LLM

**Edge-ready, multi-provider LLM orchestrator with streaming, caching, and Zod-powered structured output.**

[![NPM Version](https://img.shields.io/npm/v/universal-llm.svg)](https://www.npmjs.com/package/universal-llm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why Universal LLM?

Most LLM SDKs are heavy, Node-only, or lock you into a single provider. **Universal LLM** is built for the modern web:

- **🌍 Edge-Ready** — 100% `fetch`-based. Works on Vercel Edge, Cloudflare Workers, Bun, Deno, and the Browser.
- **🔗 Multi-Provider** — OpenAI, Anthropic, Grok (xAI), and Ollama with a single, unified API.
- **✅ Zod-Powered** — Guaranteed structured output. Define a schema, get a typed object back.
- **📡 Streaming** — Native SSE streaming for all providers via `AsyncGenerator`.
- **💾 Caching** — Built-in in-memory response cache with configurable TTL.
- **🔄 Retries** — Automatic exponential backoff on failures.
- **🪝 Middleware** — `onRequest` / `onResponse` / `onError` hooks for logging, PII scrubbing, or transformation.
- **🪶 Ultra-Lightweight** — Zero runtime dependencies (only `zod` as a peer dependency).

---

## Installation

```bash
npm install universal-llm zod
```

## Quick Start

### Simple Chat

```typescript
import { UniversalLLM } from 'universal-llm';

const llm = new UniversalLLM({
  provider: 'openai',
  apiKey: process.env.OPENAI_API_KEY,
});

const response = await llm.chat([
  { role: 'user', content: 'What is the future of AI?' }
]);

console.log(response);
```

### Switch Provider in One Line

```typescript
const llm = new UniversalLLM({
  provider: 'anthropic', // or 'grok', 'ollama'
  apiKey: process.env.ANTHROPIC_API_KEY,
});
```

### Structured Output (Zod)

Define your data shape and get a **fully-typed, validated** JavaScript object back.

```typescript
import { z } from 'zod';
import { UniversalLLM } from 'universal-llm';

const UserSchema = z.object({
  name: z.string(),
  age: z.number(),
  interests: z.array(z.string()),
});

const llm = new UniversalLLM({ provider: 'openai', apiKey: '...' });

const user = await llm.structured({
  schema: UserSchema,
  prompt: 'Extract: John is 25, loves surfing and AI.',
});

// user is fully typed as { name: string; age: number; interests: string[] }
console.log(user.name); // "John"
```

### Streaming

```typescript
const llm = new UniversalLLM({ provider: 'openai', apiKey: '...' });

for await (const chunk of llm.stream([
  { role: 'user', content: 'Write a haiku about code.' }
])) {
  process.stdout.write(chunk); // Real-time output
}
```

### Caching

```typescript
const llm = new UniversalLLM({
  provider: 'openai',
  apiKey: '...',
  cache: true,
  cacheTTL: 60000, // 1 minute
});

// First call hits the API
await llm.chat([{ role: 'user', content: 'Hello' }]);

// Second identical call returns from cache instantly
await llm.chat([{ role: 'user', content: 'Hello' }]);

// Manually clear if needed
llm.clearCache();
```

### Retries with Exponential Backoff

```typescript
const llm = new UniversalLLM({
  provider: 'openai',
  apiKey: '...',
  retries: 3,       // Retry up to 3 times
  retryDelay: 1000,  // Start with 1s, then 2s, then 4s
  timeout: 30000,    // Abort after 30 seconds
});
```

### Middleware Hooks

```typescript
const llm = new UniversalLLM({
  provider: 'openai',
  apiKey: '...',
  onRequest: (payload) => {
    console.log('Sending:', payload);
    return payload; // Optionally modify
  },
  onResponse: (text) => {
    console.log('Received:', text.length, 'chars');
    return text; // Optionally transform
  },
  onError: (err) => {
    console.error('LLM Error:', err.message);
  },
});
```

---

## API Reference

### `new UniversalLLM(config)`

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `provider` | `'openai' \| 'anthropic' \| 'grok' \| 'ollama'` | *required* | LLM provider |
| `apiKey` | `string` | `''` | API key (not needed for Ollama) |
| `baseUrl` | `string` | Provider default | Custom API endpoint |
| `timeout` | `number` | `30000` | Request timeout in ms |
| `retries` | `number` | `2` | Number of retry attempts |
| `retryDelay` | `number` | `1000` | Initial retry delay in ms |
| `cache` | `boolean` | `false` | Enable response caching |
| `cacheTTL` | `number` | `300000` | Cache time-to-live in ms |
| `onRequest` | `function` | — | Middleware before sending |
| `onResponse` | `function` | — | Middleware after receiving |
| `onError` | `function` | — | Error handler hook |

### Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `chat(messages, options?)` | `Promise<string>` | Standard completion |
| `stream(messages, options?)` | `AsyncGenerator<string>` | Streaming completion |
| `structured(options)` | `Promise<z.infer<T>>` | Zod-validated structured output |
| `clearCache()` | `void` | Clear the response cache |

### Supported Providers

| Provider | Default Model | Streaming | Edge Support |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `gpt-4o` | ✅ | ✅ |
| **Anthropic** | `claude-sonnet-4-20250514` | ✅ | ✅ |
| **Grok (xAI)** | `grok-3` | ✅ | ✅ |
| **Ollama** | `llama3` | ✅ | ✅ (Local) |

---

## License

MIT © Antigravity
