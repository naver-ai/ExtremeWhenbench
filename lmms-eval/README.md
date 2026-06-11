# ExtremeWhenBench × lmms-eval

Drop-in task plugin for [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

```
lmms-eval/
├── extremewhenbench/        # task plugin: yaml + utils + README
└── patches/chat_openai.py   # patched openai adapter (small +49-line change)
```

## Install

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval

cp -r ../ExtremeWhenBench/lmms-eval/extremewhenbench  lmms_eval/tasks/
cp    ../ExtremeWhenBench/lmms-eval/patches/chat_openai.py \
      lmms_eval/models/chat/openai.py

pip install -e .
```

The `chat_openai.py` patch adds two opt-in flags
(`pass_video_url`, `enable_thinking_kwarg`) so the OpenAI-compatible
adapter can send video as a URL and let vLLM decode server-side.
Without it, lmms-eval extracts frames client-side and the absolute-time
signal the model needs at hour-scale is lost (mIoU drops ~10×).

## Upstream PR (planned)

We plan to send the same two pieces upstream as a PR to
`EvolvingLMMs-Lab/lmms-eval`:

| Change                                                  | Files                                                          |
| ------------------------------------------------------- | -------------------------------------------------------------- |
| Add `pass_video_url` / `enable_thinking_kwarg` flags    | `lmms_eval/models/chat/openai.py` (+49 lines)                  |
| Add `extremewhenbench` task                             | `lmms_eval/tasks/extremewhenbench/{yaml, utils.py, README.md}` |

PR link: TBD. Until it lands, the drop-in above is the way to use the task.

## Run

```bash
export HF_HOME=/path/to/your/hf-cache

# 1) Make sure the three source corpora are cached under HF_HOME
python -c "from datasets import load_dataset; load_dataset('lmms-lab/LVBench')"
python -c "from datasets import load_dataset; load_dataset('sy1998/MLVU_dev')"
python -c "from datasets import load_dataset; load_dataset('lmms-lab/Video-MME')"

# 2) Serve the model
vllm serve <Qwen3.5-9B-checkpoint> --port 8000 \
    --served-model-name qwen3.5-9b \
    --tensor-parallel-size 8 --reasoning-parser qwen3 \
    --trust-remote-code --enforce-eager \
    --max-model-len 65536 \
    --allowed-local-media-path "$HF_HOME"

# 3) Evaluate
python -m lmms_eval \
    --model openai \
    --model_args "model=qwen3.5-9b,base_url=http://localhost:8000/v1,api_key=EMPTY,\
                  pass_video_url=True,max_frames_num=768,enable_thinking_kwarg=False,\
                  num_concurrent=32" \
    --gen_kwargs "max_new_tokens=128,temperature=0,top_p=0.8" \
    --tasks extremewhenbench \
    --batch_size 1 \
    --output_path ./logs/ewb --log_samples
```

If the source-corpus videos are missing, the task's preflight aggregates
every missing `(corpus, video_id)` pair into a single error before any
inference runs — so you find out up front, not three hours in. Override
the cache path per corpus with `EWB_LVBENCH_PATH`, `EWB_MLVU_PATH`, or
`EWB_VIDEOMME_PATH`.

## Reproducible number

Full 2,273 q, Qwen3.5-9B, `num_frames=768`, `enable_thinking=False`:
**mIoU ≈ 0.047 ± 0.005** (paper Table 4: 0.053).

See `extremewhenbench/README.md` for full task details and tuning notes.
