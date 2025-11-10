# Using Alternative LLM Providers

Since you're hitting Google Gemini rate limits, here are alternative LLM providers you can use:

## Option 1: OpenAI GPT-4

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run with GPT-4
python run_kernel_search_simple.py \
    --level 1 \
    --problem_id 1 \
    --steps 10 \
    --server_type openai \
    --model_name "gpt-4"
```

## Option 2: Anthropic Claude

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Run with Claude
python run_kernel_search_simple.py \
    --level 1 \
    --problem_id 1 \
    --steps 10 \
    --server_type anthropic \
    --model_name "claude-3-5-sonnet-20241022"
```

## Option 3: Wait for Gemini Rate Limit Reset

The error message shows: "Please retry in 51.102036282s"

Just wait ~1 minute and try again:
```bash
sleep 60
python run_kernel_search_simple.py \
    --level 1 \
    --problem_id 1 \
    --steps 10 \
    --server_type google
```

## Option 4: Use OpenAI-Compatible Local Models

If you have a local LLM server (like vLLM, Ollama, etc.):

```bash
# Set the base URL
export OPENAI_API_BASE="http://localhost:8000/v1"

python run_kernel_search_simple.py \
    --level 1 \
    --problem_id 1 \
    --steps 10 \
    --server_type openai \
    --model_name "your-local-model"
```

## Check Your API Keys

Your API keys are stored in `.env` file. Check which ones you have:

```bash
cat .env | grep -E "(OPENAI|ANTHROPIC|GOOGLE|GEMINI)"
```

## Model Recommendations for Kernel Optimization

- **Best Performance**: `gpt-4` or `claude-3-5-sonnet-20241022`
- **Fast & Cheap**: `gpt-4o-mini` or `gemini-1.5-flash`
- **Reasoning**: `o1-preview` (set `--is_reasoning_model`)

## Full Example with GPT-4

```bash
# Assuming you have OPENAI_API_KEY in .env or environment
python run_kernel_search_simple.py \
    --level 1 \
    --problem_id 1 \
    --steps 20 \
    --server_type openai \
    --model_name "gpt-4" \
    --temperature 0.7 \
    --num_drafts 3 \
    --debug_prob 0.3
```

This will:
- Generate 3 initial draft implementations
- Run for 20 search steps
- Use GPT-4 for code generation
- Have 30% probability of debugging failed kernels
