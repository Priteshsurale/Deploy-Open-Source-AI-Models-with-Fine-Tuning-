# Deploy Open-Source AI Models with Fine-Tuning (LoRA & QLoRA) and API Endpoints on Cloud

## Introduction
This document demonstrates how to deploy open-source AI models (e.g., LLaMA, Whisper, Mistral, Gemma) on cloud platforms like Cloudera, and how to create an API endpoint similar to OpenAI's API. The API will accept various types of data (image, audio, text, or combinations) and return processed responses. 

## Cloud Machine Setup

### Cloud Platform Options
You can rent cloud machines from the following platforms:
- **DigitalOcean**
- **Cloudera**
- **runpod** 
- **AWS**
- **Google Cloud**
- **Microsoft Azure**

Choose a platform based on cost and availability of GPU resources.

### Hardware and Software Requirements
- **GPU**: Recommended (e.g., NVIDIA A100, V100, 4x A40 OR 2x A100) for fine-tuning and inference.
- **RAM**: 16 GB or more (recommended).
- **Storage**: 100 GB or more.

    - The Choice of GPU 
      - Llama 3.1 70B FP16: 4x A40 or 2x A100
      - Llama 3.1 70B INT8: 1x A100 or 2x A40
      - Llama 3.1 70B INT4: 1x A40

Software:
- **Python 3.7+**
- **PyTorch**
- **Transformers** (Hugging Face)
- **BitsAndBytes** (for QLoRA)

## Open Source Model
 - Chat Completion
   - llama3.2
   - mistral
   - gemma
 
 - vision
   - llama 3.2

 - Speech To Text  
   - Wav2vec
   - whisper    

check out the specific models - [Hugging Face Model](https://huggingface.co/models) 

## Fine-Tuning the Model

### [Hugging Face- SFT TRl](https://huggingface.co/docs/trl/en/sft_trainer)
### PEFT (Parameter-Efficient Fine-Tuning)
### LoRA (Low-Rank Adaptation)
### QLoRA (Quantized Low-Rank Adaptation)
### [unsloth](https://github.com/unslothai/unsloth) 


## Reference Links
- [Self-Hosting LLaMA 3.1 70B (or any ~70B LLM) Affordably](https://abhinand05.medium.com/self-hosting-llama-3-1-70b-or-any-70b-llm-affordably-2bd323d72f8d)
- [Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
- [alignment-handbook](https://github.com/huggingface/alignment-handbook)
- [Step-by-Step Guide to Building a Speech and Voice AI Assistant Using ASR with Llama 3.1](https://www.e2enetworks.com/blog/step-by-step-guide-to-building-a-speech-and-voice-ai-assistant-using-asr-with-llama-3-1)
