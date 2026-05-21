import { describe, it, expect, vi, beforeEach } from 'vitest';
import { UniversalLLM } from './index';
import { z } from 'zod';

// ──────────────────────────────────────────────
// Mock fetch globally
// ──────────────────────────────────────────────

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

function mockOpenAIResponse(content: string) {
  return new Response(JSON.stringify({
    choices: [{ message: { content } }]
  }), { status: 200 });
}

function mockAnthropicResponse(text: string) {
  return new Response(JSON.stringify({
    content: [{ text }]
  }), { status: 200 });
}

function mockOllamaResponse(content: string) {
  return new Response(JSON.stringify({
    message: { content }
  }), { status: 200 });
}

beforeEach(() => {
  mockFetch.mockReset();
});

// ──────────────────────────────────────────────
// 1. BASIC CHAT
// ──────────────────────────────────────────────

describe('UniversalLLM.chat', () => {
  it('should call OpenAI and return content', async () => {
    mockFetch.mockResolvedValueOnce(mockOpenAIResponse('Hello from GPT'));

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test-key' });
    const result = await llm.chat([{ role: 'user', content: 'Hi' }]);

    expect(result.data).toBe('Hello from GPT');
    expect(mockFetch).toHaveBeenCalledOnce();

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe('https://api.openai.com/v1/chat/completions');
    expect(JSON.parse(init.body).model).toBe('gpt-4o');
    expect(init.headers['Authorization']).toBe('Bearer test-key');
  });

  it('should call Anthropic and return content', async () => {
    mockFetch.mockResolvedValueOnce(mockAnthropicResponse('Hello from Claude'));

    const llm = new UniversalLLM({ provider: 'anthropic', apiKey: 'test-key' });
    const result = await llm.chat([
      { role: 'system', content: 'You are helpful' },
      { role: 'user', content: 'Hi' }
    ]);

    expect(result.data).toBe('Hello from Claude');

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    // Anthropic should filter out system messages from the messages array
    expect(body.messages.length).toBe(1);
    expect(body.system).toBe('You are helpful');
  });

  it('should call Grok and return content', async () => {
    mockFetch.mockResolvedValueOnce(mockOpenAIResponse('Hello from Grok'));

    const llm = new UniversalLLM({ provider: 'grok', apiKey: 'test-key' });
    const result = await llm.chat([{ role: 'user', content: 'Hi' }]);

    expect(result.data).toBe('Hello from Grok');
    expect(mockFetch.mock.calls[0][0]).toBe('https://api.x.ai/v1/chat/completions');
  });

  it('should call Ollama and return content', async () => {
    mockFetch.mockResolvedValueOnce(mockOllamaResponse('Hello from Llama'));

    const llm = new UniversalLLM({ provider: 'ollama' });
    const result = await llm.chat([{ role: 'user', content: 'Hi' }]);

    expect(result.data).toBe('Hello from Llama');
    expect(mockFetch.mock.calls[0][0]).toBe('http://localhost:11434/api/chat');

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.stream).toBe(false);
  });

  it('should use custom baseUrl when provided', async () => {
    mockFetch.mockResolvedValueOnce(mockOpenAIResponse('Custom'));

    const llm = new UniversalLLM({
      provider: 'openai',
      apiKey: 'test',
      baseUrl: 'https://my-proxy.com/v1/chat/completions'
    });
    await llm.chat([{ role: 'user', content: 'test' }]);

    expect(mockFetch.mock.calls[0][0]).toBe('https://my-proxy.com/v1/chat/completions');
  });

  it('should use custom model when provided', async () => {
    mockFetch.mockResolvedValueOnce(mockOpenAIResponse('OK'));

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test' });
    await llm.chat([{ role: 'user', content: 'Hi' }], { model: 'gpt-3.5-turbo' });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.model).toBe('gpt-3.5-turbo');
  });
});

// ──────────────────────────────────────────────
// 2. ERROR HANDLING
// ──────────────────────────────────────────────

describe('Error handling', () => {
  it('should throw on API error with message', async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ error: { message: 'Invalid API key' } }), { status: 401, statusText: 'Unauthorized' })
    );

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'bad-key', retries: 0 });

    await expect(llm.chat([{ role: 'user', content: 'Hi' }]))
      .rejects.toThrow('Invalid API key');
  });

  it('should throw on unknown provider', async () => {
    const llm = new UniversalLLM({ provider: 'unknown' as any, retries: 0 });

    await expect(llm.chat([{ role: 'user', content: 'Hi' }]))
      .rejects.toThrow('Unknown provider');
  });
});

// ──────────────────────────────────────────────
// 3. RETRIES
// ──────────────────────────────────────────────

describe('Retry logic', () => {
  it('should retry on failure and succeed', async () => {
    mockFetch
      .mockRejectedValueOnce(new Error('Network error'))
      .mockResolvedValueOnce(mockOpenAIResponse('Recovered'));

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test', retries: 2, retryDelay: 10 });
    const result = await llm.chat([{ role: 'user', content: 'Hi' }]);

    expect(result.data).toBe('Recovered');
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it('should exhaust retries and throw', async () => {
    mockFetch.mockRejectedValue(new Error('Persistent failure'));

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test', retries: 1, retryDelay: 10 });

    await expect(llm.chat([{ role: 'user', content: 'Hi' }]))
      .rejects.toThrow('Persistent failure');
    expect(mockFetch).toHaveBeenCalledTimes(2); // initial + 1 retry
  });

  it('should call onError hook for each failure', async () => {
    const onError = vi.fn();
    mockFetch
      .mockRejectedValueOnce(new Error('Fail 1'))
      .mockResolvedValueOnce(mockOpenAIResponse('OK'));

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test', retries: 1, retryDelay: 10, onError });
    await llm.chat([{ role: 'user', content: 'Hi' }]);

    expect(onError).toHaveBeenCalledOnce();
    expect(onError.mock.calls[0][0].message).toBe('Fail 1');
  });
});

// ──────────────────────────────────────────────
// 4. CACHING
// ──────────────────────────────────────────────

describe('Caching', () => {
  it('should return cached response on second call', async () => {
    mockFetch.mockResolvedValueOnce(mockOpenAIResponse('Cached result'));

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test', cache: true });
    const messages = [{ role: 'user' as const, content: 'What is 2+2?' }];

    const result1 = await llm.chat(messages);
    const result2 = await llm.chat(messages);

    expect(result1.data).toBe('Cached result');
    expect(result2.data).toBe('Cached result');
    expect(mockFetch).toHaveBeenCalledOnce(); // Only 1 API call
  });

  it('should not cache when cache is disabled', async () => {
    mockFetch
      .mockResolvedValueOnce(mockOpenAIResponse('First'))
      .mockResolvedValueOnce(mockOpenAIResponse('Second'));

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test', cache: false, retries: 0 });
    const messages = [{ role: 'user' as const, content: 'Hi' }];

    const r1 = await llm.chat(messages);
    const r2 = await llm.chat(messages);

    expect(r1.data).toBe('First');
    expect(r2.data).toBe('Second');
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it('should clear cache when clearCache() is called', async () => {
    mockFetch
      .mockResolvedValueOnce(mockOpenAIResponse('Before'))
      .mockResolvedValueOnce(mockOpenAIResponse('After'));

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test', cache: true, retries: 0 });
    const messages = [{ role: 'user' as const, content: 'Hi' }];

    await llm.chat(messages);
    llm.clearCache();
    const result = await llm.chat(messages);

    expect(result.data).toBe('After');
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });
});

// ──────────────────────────────────────────────
// 5. MIDDLEWARE
// ──────────────────────────────────────────────

describe('Middleware hooks', () => {
  it('should call onRequest before sending', async () => {
    mockFetch.mockResolvedValueOnce(mockOpenAIResponse('OK'));

    const onRequest = vi.fn((payload) => {
      return { ...payload, model: 'gpt-3.5-turbo' };
    });

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test', onRequest });
    await llm.chat([{ role: 'user', content: 'Hi' }]);

    expect(onRequest).toHaveBeenCalledOnce();

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.model).toBe('gpt-3.5-turbo');
  });

  it('should call onResponse after receiving', async () => {
    mockFetch.mockResolvedValueOnce(mockOpenAIResponse('raw response'));

    const onResponse = vi.fn((text: string) => text.toUpperCase());

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test', onResponse });
    const result = await llm.chat([{ role: 'user', content: 'Hi' }]);

    expect(result.data).toBe('RAW RESPONSE');
    expect(onResponse).toHaveBeenCalledWith('raw response');
  });
});

// ──────────────────────────────────────────────
// 6. STRUCTURED OUTPUT
// ──────────────────────────────────────────────

describe('Structured output', () => {
  it('should parse JSON response into Zod schema', async () => {
    const schema = z.object({
      name: z.string(),
      age: z.number(),
    });

    mockFetch.mockResolvedValueOnce(
      mockOpenAIResponse('{"name": "Alice", "age": 30}')
    );

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test' });
    const result = await llm.structured({
      schema,
      prompt: 'Extract info from: Alice is 30 years old.',
    });

    expect(result.data.name).toBe('Alice');
    expect(result.data.age).toBe(30);
  });

  it('should handle markdown-fenced JSON response', async () => {
    const schema = z.object({ color: z.string() });

    mockFetch.mockResolvedValueOnce(
      mockOpenAIResponse('```json\n{"color": "blue"}\n```')
    );

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test' });
    const result = await llm.structured({
      schema,
      prompt: 'What color is the sky?',
    });

    expect(result.data.color).toBe('blue');
  });

  it('should throw when response does not match schema', async () => {
    const schema = z.object({ name: z.string(), age: z.number() });

    mockFetch.mockResolvedValueOnce(
      mockOpenAIResponse('{"name": "Bob"}') // Missing required "age"
    );

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test', retries: 0 });

    await expect(llm.structured({
      schema,
      prompt: 'Extract info',
    })).rejects.toThrow('Structured output failed');
  });

  it('should throw when response is not valid JSON', async () => {
    const schema = z.object({ x: z.number() });

    mockFetch.mockResolvedValueOnce(
      mockOpenAIResponse('This is not JSON at all.')
    );

    const llm = new UniversalLLM({ provider: 'openai', apiKey: 'test', retries: 0 });

    await expect(llm.structured({
      schema,
      prompt: 'Give me a number',
    })).rejects.toThrow('Structured output failed');
  });
});
