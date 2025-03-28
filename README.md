# Advanced Optimization Techniques for Linux Systems: Running Large Language Models (LLMs) - Revised Procedures Guide - Version 1.1 

.

## 1: Introduction - Optimizing LLMs on Linux



1.1 Welcome and Purpose of the Guide

1.2 Target Audience and Skill Level

1.3 Defining "Resource-Constrained" Systems

1.4 Importance of Optimization on Linux

1.5 Overview of Optimization Techniques Covered

1.6 Understanding Trade-offs (Speed, Memory, Accuracy)


.

    
## 2: Understanding Memory Constraints for LLMs



2.1 VRAM (Video RAM) vs. System RAM

2.2 Importance of VRAM for LLM Inference

2.3 Factors Affecting Memory Usage: Model Size, Precision, Context Length

2.4 Monitoring Memory Usage on Linux (e.g., nvidia-smi, free -m)

.
    
## 3: Quantization Techniques - Reducing Model Footprint


3.1 Introduction to Quantization

3.2 Why Quantization Works (Reduced Precision, Smaller Model Size)

3.3 Types of Quantization: Post-Training Quantization (PTQ), Quantization-Aware Training (QAT)

3.4 Focus on PTQ for Inference Optimization

.
    
## 4: GPTQ and ExLlamaV2 - High-Performance Quantization


4.1 Introduction to GPTQ (Generative Post-training Quantization)

4.2 ExLlamaV2 Library for Fast GPTQ Inference

4.3 Advantages of GPTQ: Speed, Compression, NVIDIA GPU Optimization

4.4 Trade-offs: Potential Accuracy Loss

4.5 Recipe: Running GPTQ Quantized Models with ExLlamaV2 - Step-by-Step

4.6 Troubleshooting GPTQ Loading: Common Errors and Solutions

4.7 Practical Example with AutoGPTQ and Transformers (Python Code)

.
    
## 5: GGML/GGUF and llama.cpp - CPU and Cross-Platform Efficiency


5.1 Introduction to GGML/GGUF Model Format

5.2 llama.cpp Library: CPU-Optimized Inference

5.3 Versatility: CPU and GPU Support, Various Quantization Levels

5.4 Cross-Platform Compatibility (Linux, macOS, Windows)

5.5 Recipe: Running GGUF Models with llama.cpp - Step-by-Step

5.6 Troubleshooting llama.cpp Build Issues: Common Compiler Errors and Dependencies

5.7 Practical Example: Building llama.cpp and Running Inference (Bash Commands)

.
    
## 6: Bitsandbytes - Easy Quantization in Transformers


6.1 Introduction to Bitsandbytes Library

6.2 Ease of Integration with Hugging Face Transformers

6.3 Support for 4-bit and 8-bit Quantization

6.4 Mixed-Precision Strategies

6.5 Use Cases: Quick Experimentation, Python Workflows, Memory-Efficient Training

6.6 Recipe: Loading a 4-bit Quantized Model with Bitsandbytes - Step-by-Step

6.7 Practical Example: Loading a 4-bit Quantized Model with Transformers (Python Code)

.
    
## 7: Model Selection - Choosing Efficient LLM Architectures


7.1 Importance of Model Architecture for Efficiency

7.2 Choosing Smaller Models vs. Larger Models

7.3 Distilled/Optimized Models (e.g., MobileBERT, DistilBERT, Smaller LLM Variants)

7.4 Tokenizer Efficiency (SentencePiece, Tiktoken, BPE)

7.5 Specific Model Recommendations for Resource-Constrained Systems

7.6 Using Hugging Face Hub Filters to Find Efficient Models

7.7 Model Selection Criteria Beyond Size: Task Suitability and Architecture

.
    
## 8: Offloading to System RAM and NVMe - Expanding Memory Capacity


8.1 Concept of Model Offloading

8.2 VRAM Overflow and Necessity of Offloading

8.3 Offloading to System RAM: Speed Trade-off

8.4 Offloading to NVMe SSD: Better Performance than HDD, Still Slower than VRAM

8.5 Recipe: Model Offloading with `accelerate` - Step-by-Step

8.6 Monitoring I/O Performance with iostat

8.7 Using accelerate Library for Model Offloading (Conceptual Example)

.
    
## 9: Memory Mapping (mmap) for Efficient Model Loading


9.1 Concept of Memory Mapping (mmap)

9.2 Benefits of mmap for LLM Loading: Speed, Memory Efficiency

9.3 How mmap Works with Model Files (e.g., .safetensors, .gguf)

9.4 Practical Considerations: File System Caching, Shared Memory

9.5 Python Libraries Leveraging mmap (e.g., `safetensors` library)

.
    
## 10: Compilation and Graph Optimization - Speeding Up Inference


10.1 Just-In-Time (JIT) Compilation for LLMs

10.2 Graph Optimization Techniques (Operator Fusion, Kernel Optimization)

10.3 Libraries for Compilation and Optimization (e.g., TorchScript, ONNX Runtime, TensorRT, DeepSpeed)

10.4 Trade-offs: Compilation Time vs. Inference Speedup, Hardware Compatibility

10.5 Conceptual Examples and Tools

10.6 TorchScript Example (PyTorch Code)

.
    
## 11: Hardware Acceleration - Leveraging GPUs and Specialized Hardware


11.1 Benefits of GPUs for LLM Inference

11.2 NVIDIA GPUs and CUDA

11.3 AMD GPUs and ROCm

11.4 Apple Silicon GPUs and Metal

11.5 CPUs with Vector Extensions (AVX-512, AVX2)

11.6 Specialized AI Accelerators (Conceptual Overview - TPUs, Inferentia, etc.)

11.7 Choosing the Right Hardware for Your Budget and Performance Needs - Specific Recommendations

.
    
## 12: Conclusion - Summary and Future Directions


12.1 Recap of Optimization Techniques Covered

12.2 Key Takeaways and Best Practices

12.3 Future Trends in LLM Optimization

12.4 Community Resources and Further Learning

12.5 Final Words

.

## **1: Introduction - Optimizing LLMs on Linux**


### **1.1 Welcome and Purpose of the Guide**


Welcome to this revised guide on advanced optimization techniques for running Large Language Models (LLMs) on Linux systems. This guide is designed for developers and researchers who want to leverage the power of LLMs even on systems with limited resources.  This version 1.1 includes step-by-step recipes, troubleshooting tips, and more detailed hardware recommendations based on user feedback.


### **1.2 Target Audience and Skill Level**


This guide is intended for developers and researchers with some familiarity with Linux, Python, and basic machine learning concepts.  While advanced deep learning expertise is not required, a basic understanding of neural networks and the Hugging Face Transformers library will be beneficial.


### **1.3 Defining "Resource-Constrained" Systems**


We understand that not everyone has access to high-end servers with multiple top-tier GPUs. This guide focuses on practical methods to run and fine-tune LLMs effectively on resource-constrained Linux environments, such as laptops, desktops with consumer-grade GPUs, or budget-friendly cloud instances. By "resource-constrained," we specifically mean systems that might have limitations in:


*   **VRAM (Video RAM):** Less than 24GB (e.g., 6GB, 8GB, 12GB, 16GB).

*   **System RAM:** Less than 32GB (e.g., 8GB, 16GB, 24GB).

*   **CPU Cores:** Less than 8 cores (e.g., 2 cores, 4 cores, 6 cores).


### **1.4 Importance of Optimization on Linux**


Linux is a powerful and flexible operating system, making it an excellent platform for LLM experimentation and deployment. However, running large models efficiently requires careful optimization.  Without optimization, you might encounter:


*   **Out-of-Memory Errors:**  Models exceeding available VRAM or system RAM.

*   **Slow Inference Speed:**  Unacceptably slow text generation.

*   **System Unresponsiveness:**  Overloading system resources.


This guide will walk you through a range of techniques, from model quantization and selection to hardware acceleration and advanced memory management, all within the Linux environment.


### **1.5 Overview of Optimization Techniques Covered**


This guide covers the following key optimization techniques:


*   **Quantization:** Reducing model precision (GPTQ, GGUF, bitsandbytes).

*   **Model Selection:** Choosing smaller and more efficient models.

*   **Memory Management:** Offloading and memory mapping.

*   **Compilation and Graph Optimization:** Speeding up inference with tools like TorchScript and TensorRT.

*   **Hardware Acceleration:** Leveraging GPUs and CPU vector extensions.


### **1.6 Understanding Trade-offs (Speed, Memory, Accuracy)**


It's crucial to remember that optimization often involves trade-offs. Techniques that save memory or boost speed might sometimes slightly affect the model's accuracy.  For example:


*   **Quantization:** Lower precision can reduce accuracy, but the impact is often minimal, especially with techniques like GPTQ and GGUF.

*   **Model Selection:** Smaller models are faster and use less memory but might have lower overall performance than larger models on complex tasks.

*   **Offloading:** Offloading to system RAM or NVMe allows running larger models but slows down inference speed.


This guide will highlight these trade-offs, empowering you to make informed choices tailored to your specific needs and priorities. We will cover techniques to optimize for both memory footprint and computational speed, enabling you to run impressive LLMs even on modest hardware. Let's begin our journey into the world of LLM optimization on Linux!


---


## **2: Understanding Memory Constraints for LLMs**


### **2.1 VRAM (Video RAM) vs. System RAM**


Before diving into optimization techniques, it's essential to understand the memory landscape when running LLMs. Two types of memory are particularly relevant: VRAM (Video RAM) and System RAM.


*   **VRAM (Video RAM):**  Memory directly attached to your GPU (Graphics Processing Unit).  For GPU-accelerated LLM inference, VRAM is the primary bottleneck.  LLMs require significant VRAM to store:

    *   **Model Weights:** The parameters of the LLM.

    *   **Activations:** Intermediate calculations during inference.

    *   **KV Cache:**  For efficient attention mechanisms, storing key-value pairs from previous tokens.

    If your model and its working data exceed your GPU's VRAM capacity, you will encounter "out-of-memory" errors, preventing successful inference.


*   **System RAM:** The main memory of your computer, used by the CPU and all running processes. While less critical than VRAM for GPU-accelerated inference, system RAM becomes important when:

    *   **Offloading:** When VRAM is insufficient, techniques like model offloading move parts of the model to system RAM (or even slower storage like NVMe).

    *   **CPU Inference:** If you are running inference on the CPU (e.g., using llama.cpp without GPU acceleration), the model and computations will reside in system RAM.

    *   **Data Handling:** Tokenization, pre-processing, and post-processing of text data also consume system RAM.


### **2.2 Importance of VRAM for LLM Inference**


VRAM is often the most critical resource for running LLMs, especially when using GPUs for acceleration.  Running out of VRAM is a common problem and the primary driver for many optimization techniques.  Efficient VRAM usage allows you to:


*   Run larger, more capable models.

*   Increase batch sizes for higher throughput.

*   Use longer context lengths for more complex prompts and generations.


### **2.3 Factors Affecting Memory Usage: Model Size, Precision, Context Length**


Several factors influence the memory footprint of an LLM:


*   **Model Size (Number of Parameters):** Larger models with more parameters naturally require more memory.  Model size is typically measured in billions of parameters (e.g., 7B, 13B, 70B, 175B).

*   **Precision (Data Type):**

    *   **Full Precision (FP32):** 32-bit floating-point, consumes the most memory.

    *   **Half Precision (FP16, BF16):** 16-bit floating-point, reduces memory usage by half compared to FP32. BF16 (BFloat16) is often preferred for training and some inference scenarios.

    *   **Integer Quantization (INT8, INT4):** 8-bit or 4-bit integers, significantly reduce memory usage (e.g., INT4 can be 8x smaller than FP32).

*   **Context Length:** Longer input sequences (prompts) and longer generated outputs increase the size of the KV cache, consuming more VRAM during inference.


### **2.4 Monitoring Memory Usage on Linux (e.g., nvidia-smi, free -m)**


Monitoring memory usage is crucial for understanding your system's resource utilization and diagnosing memory-related issues.  On Linux, you can use the following commands:


*   **GPU VRAM (NVIDIA):** Use the `nvidia-smi` command in your terminal.

    ```

    nvidia-smi

    ```

    This provides real-time information about GPU utilization, memory usage, temperature, and more. Look for the "Used Memory" column for your GPU.


*   **GPU VRAM (AMD):** Use `rocm-smi` for AMD GPUs with ROCm.

    ```

    rocm-smi

    ```


*   **System RAM:** Use the `free -m` command to display system RAM usage in megabytes, or `free -g` for gigabytes.

    ```

    free -m

    ```

    or

    ```

    free -g

    ```


Understanding these memory constraints is the first step towards effective optimization. In the following chapters, we will explore techniques to reduce memory usage and improve performance within these limitations.


---


## **3: Quantization Techniques - Reducing Model Footprint**


### **3.1 Introduction to Quantization**


Quantization is a cornerstone technique for reducing the memory footprint and accelerating the inference speed of LLMs. It works by reducing the numerical precision of the model's weights and sometimes activations.  Think of it like compressing an image - you reduce the number of bits used to represent colors, making the file smaller and potentially faster to load and display, sometimes with a slight loss of visual fidelity.


### **3.2 Why Quantization Works (Reduced Precision, Smaller Model Size)**


LLMs are typically trained using high-precision floating-point numbers (like FP32 or FP16) to represent their parameters (weights). However, for inference, especially on resource-constrained devices, this high precision is often not strictly necessary. Quantization converts these high-precision weights into lower-precision integer or lower-bit floating-point formats (e.g., INT8, INT4, FP16, BF16).


*   **Reduced Precision, Smaller Model Size:** Lower precision means fewer bits are used to represent each parameter. For example:

    *   FP32 (32 bits per weight)

    *   FP16 (16 bits per weight) - 2x reduction

    *   INT8 (8 bits per weight) - 4x reduction

    *   INT4 (4 bits per weight) - 8x reduction!

    Converting from FP32 to INT4 can potentially reduce model size by a factor of 8! This allows larger models to fit in GPU VRAM (Video RAM) or system RAM.


*   **Faster Computation:** Integer and lower-precision floating-point operations are often significantly faster than full-precision floating-point operations, especially on modern hardware with specialized instructions for these data types (e.g., Tensor Cores on NVIDIA GPUs are optimized for lower precision).


### **3.3 Types of Quantization: Post-Training Quantization (PTQ), Quantization-Aware Training (QAT)**


There are two main categories of quantization:


*   **Post-Training Quantization (PTQ):** This is the most common and easiest type of quantization for inference optimization. PTQ is applied *after* a model has been fully trained. It typically involves converting the weights of a pre-trained model to a lower precision format. PTQ can be further categorized into:

    *   **Weight-Only Quantization:** Only the model weights are quantized, while activations and computations might remain in higher precision (mixed-precision). GPTQ and GGML/GGUF are examples of weight-only quantization techniques. This is generally faster and easier to implement.

    *   **Weight and Activation Quantization:** Both weights and activations are quantized. This can provide further memory and speed benefits but is generally more complex and might require more careful calibration to minimize accuracy loss.


*   **Quantization-Aware Training (QAT):** This is a more advanced technique where quantization is incorporated *during* the training process itself. QAT typically leads to better accuracy compared to PTQ at very low bitwidths because the model learns to compensate for the quantization effects during training. However, QAT requires retraining the model, which is computationally expensive and often not feasible for end-users of pre-trained models.


### **3.4 Focus on PTQ for Inference Optimization**


For the purpose of running LLMs on resource-constrained Linux systems, we will primarily focus on Post-Training Quantization (PTQ) techniques. PTQ offers a good balance of ease of use, memory savings, and inference speed improvements without requiring model retraining. In the following chapters, we will explore specific PTQ methods like GPTQ, GGML/GGUF, and bitsandbytes in detail.


---
ExLlamaV2 Procedures Manual: Model Conversion and Evaluation


This manual outlines the procedures for converting models to the EXL2
 format and evaluating them using the provided ExLlamaV2 scripts, based 
on the documentation provided.


1. Prerequisites:


Before you begin, ensure you have the following:


Hardware:

Sufficient RAM:  At least 16GB for 7B models, and 64GB for 70B models (or larger).
NVIDIA GPU with sufficient VRAM: At least 8GB for 
7B models, and 24GB for 70B models (or larger). Mixtral 8x7B requires 
approximately 20GB of VRAM.  Ampere, Ada Lovelace, or newer 
architectures (RTX 3000/4000 series and later) are recommended for 
optimal performance with ExLlamaV2.


Software:

Python Environment: Python 3.x is required. It's recommended to use a virtual environment to manage dependencies.
Required Python Packages: Install the necessary packages using pip:

pip install auto-gptq optimum transformers (For general GPTQ related functionalities, might be implicitly needed for EXL2 conversion)
pip install human-eval (For HumanEval evaluation)
pip install datasets (For MMLU evaluation)






2. Model Conversion to EXL2 Format (using convert.py):


This section guides you through converting a pre-trained FP16 model (in Hugging Face format) to the EXL2 quantized format.


Step 2.1: Prepare your FP16 Model:


Ensure you have the source FP16 model in a directory. This directory should contain:

config.json: Model configuration file.
tokenizer.model: Tokenizer model file.
One or more .safetensors files: Model weights files (sharded models are supported).




Step 2.2: Choose a Working Directory and Output Directory:


Working Directory (-o or --out_dir):
  Select a directory for temporary files and the initial output. If you 
interrupt the conversion, you can resume from this directory.
Output Directory (-cf or --compile_full - Optional):  Choose a directory where you want to save the final quantized model in a complete, ready-to-use format. If -cf is specified, the quantized .safetensors files and all other original model files (except original FP16 .safetensors) will be copied to this directory. If -cf is not specified, the quantized .safetensors files will be saved in the working directory.


Step 2.3: Perform the Measurement Pass (Recommended):


The measurement pass analyzes the model to determine optimal 
quantization parameters. It's recommended to perform this pass first, 
especially if you plan to quantize the same model at different bitrates.




Run the convert.py script with the following arguments:


Bash


python convert.py \

    -i <path_to_fp16_model_directory> \

    -o <path_to_working_directory> \

    -nr \

    -om <path_to_measurement.json>


-i <path_to_fp16_model_directory>: Replace with the path to your FP16 model directory.
-o <path_to_working_directory>: Replace with the path to your chosen working directory.
-nr or --no_resume:  Ensures a fresh start, deleting any existing files in the working directory.
-om <path_to_measurement.json> or --output_measurement <path_to_measurement.json>:  Specifies the path to save the measurement.json file. This file will contain the measurement results.




Wait for the Measurement Pass to Complete: This pass can take a significant amount of time as it effectively quantizes the model multiple times.




Step 2.4: Perform the Quantization Pass:


Now, use the measurement results to quantize the model to your desired average bitrate.




Run the convert.py script with the following arguments:


Bash


python convert.py \

    -i <path_to_fp16_model_directory> \

    -o <path_to_working_directory> \

    -nr \

    -m <path_to_measurement.json> \

    -cf <path_to_output_directory> \

    -b <target_average_bits_per_weight>


-i <path_to_fp16_model_directory>: Replace with the path to your FP16 model directory.
-o <path_to_working_directory>: Replace with the path to your working directory (same as in the measurement pass).
-nr or --no_resume: Ensures a fresh start.
-m <path_to_measurement.json> or --measurement <path_to_measurement.json>:  Specifies the path to the measurement.json file generated in the previous step. This skips the measurement pass and uses the existing results.
-cf <path_to_output_directory> or --compile_full <path_to_output_directory>: Replace with the path to your desired output directory.
-b <target_average_bits_per_weight> or --bits <target_average_bits_per_weight>: Replace with the target average bits per weight you want to achieve (e.g., 3.0, 4.0, 2.55).




Wait for the Quantization Pass to Complete: This pass will be faster than the measurement pass if you used the -m argument.




Example: Converting Llama 2 7B to EXL2 at 3.0 bits per weight:


Assuming your FP16 Llama 2 7B model is in /mnt/models/llama2-7b-fp16/, you want to use /mnt/temp/exl2/ as the working directory, and save the final EXL2 model to /mnt/models/llama2-7b-exl2/3.0bpw/:


Measurement Pass:


Bash


python convert.py \

    -i /mnt/models/llama2-7b-fp16/ \

    -o /mnt/temp/exl2/ \

    -nr \

    -om /mnt/models/llama2-7b-exl2/measurement.json


Quantization Pass:


Bash


python convert.py \

    -i /mnt/models/llama2-7b-fp16/ \

    -o /mnt/temp/exl2/ \

    -nr \

    -m /mnt/models/llama2-7b-exl2/measurement.json \

    -cf /mnt/models/llama2-7b-exl2/3.0bpw/ \

    -b 3.0


Step 2.5: Resuming an Interrupted Job:


If the conversion process is interrupted, you can resume it by simply running the convert.py script with the working directory specified using -o:


Bash


python convert.py -o <path_to_working_directory>


Important Notes for Conversion:


-nr flag: Use -nr when starting a new conversion or if you want to clear the working directory. Omit -nr to resume an existing job.
-om and -m flags:  Using -om to save the measurement and -m to reuse it is highly recommended for efficiency, especially when experimenting with different bitrates for the same model.
"Solving..." Step: If the conversion seems to pause
 at the "Solving..." step, be patient. The script is searching for 
optimal quantization parameters, which can be computationally intensive.
MoE Models Warning:  Warnings about "less than 10% 
calibration" for certain experts in MoE models might appear. These 
warnings may or may not be critical, and further investigation might be 
needed for MoE model quantization.
Calibration Perplexity: After conversion, check the
 "calibration perplexity (quant)" in the output. A very high perplexity 
(30+) suggests potential quantization issues, and extremely high values 
(thousands) indicate a catastrophic failure.


3. Model Evaluation:


This section describes how to evaluate your EXL2 quantized models using HumanEval and MMLU benchmarks.


3.1: HumanEval Evaluation:




Ensure human-eval is installed: pip install human-eval




Run the eval/humaneval.py script:


Bash


python eval/humaneval.py \

    -m <path_to_exl2_model_directory> \

    -o humaneval_output.json


-m <path_to_exl2_model_directory> or --model <path_to_exl2_model_directory>: Replace with the path to your EXL2 model directory (the one created using -cf in the conversion process).
-o humaneval_output.json or --output humaneval_output.json: Specifies the output JSON file to store generated samples.






Evaluate Functional Correctness: Use the evaluate-functional-correctness script from the human-eval package:


Bash


evaluate-functional-correctness humaneval_output.json




Optional HumanEval Arguments:


-spt <int> or --samples_per_task <int>:  Number of samples per task (default is 1, more samples for more accurate scores).
--max_tokens <int>: Maximum tokens to generate per sample (default is 768).
-pf <str> or --prompt <str>:  Prompt format for instruct-tuned models.
-v or --verbose: Output completions during generation.
-cs <int> or --cache_size <int>: Cache size in tokens for batching performance.
-cq4, -cq6, -cq8: Use Q4, Q6, or Q8 cache respectively.


3.2: MMLU Evaluation:




Ensure datasets is installed: pip install datasets




Run the eval/mmlu.py script:


Bash


python eval/mmlu.py -m <path_to_exl2_model_directory>


-m <path_to_exl2_model_directory> or --model <path_to_exl2_model_directory>: Replace with the path to your EXL2 model directory.




Optional MMLU Arguments:


-sub <list> or --subjects <list>: Limit evaluation to specific subjects (comma-separated list).
-fs <int> or --fewshot_examples <int>: Number of few-shot examples (default is 5).
-shf or --shuffle: Shuffle answer choices.
-cs <int> or --cache_size <int>: Cache size in tokens.
-cq4, -cq6, -cq8: Use Q4, Q6, or Q8 cache.


4. Troubleshooting:


"CUDA out of memory" Errors: Reduce context length (-l in evaluation scripts) or use a smaller model. Ensure no other VRAM-intensive applications are running.
ModuleNotFoundError: Double-check that you have installed all required Python packages using pip install ....
Model Loading Errors: Verify that the model directory path (-m) is correct and that the model files are present and not corrupted.

## **4: GPTQ and ExLlamaV2 - High-Performance Quantization**


### **4.1 Introduction to GPTQ (Generative Post-training Quantization)**


GPTQ (Generative Post-training Quantization) is a powerful post-training quantization technique specifically designed for transformer-based models like LLMs. It offers excellent compression ratios and fast inference speeds, particularly when combined with the ExLlamaV2 library. GPTQ focuses on quantizing model weights to very low precision, typically 4-bit, while aiming to minimize accuracy loss.


### **4.2 ExLlamaV2 Library for Fast GPTQ Inference**


ExLlamaV2 is a highly optimized inference library specifically designed to accelerate inference with GPTQ-quantized models, especially on NVIDIA GPUs (Ampere, Ada Lovelace architectures and newer - RTX 3000/4000 series and later). It leverages highly optimized CUDA kernels for fast matrix multiplications and other operations, making GPTQ models run significantly faster than with generic inference libraries.

1. Model Conversion (convert.py) Examples:


Example 1: Basic Conversion


Bash


python convert.py \

    -i /mnt/models/llama2-7b-fp16/ \

    -o /mnt/temp/exl2/ \

    -cf /mnt/models/llama2-7b-exl2/3.0bpw/ \

    -b 3.0


Example 2: Measurement Pass Only


Bash


python convert.py \

    -i /mnt/models/llama2-7b-fp16/ \

    -o /mnt/temp/exl2/ \

    -nr \

    -om /mnt/models/llama2-7b-exl2/measurement.json


Example 3: Quantization using Measurement File (Two Bitrates)


Bash


python convert.py \

    -i /mnt/models/llama2-7b-fp16/ \

    -o /mnt/temp/exl2/ \

    -nr \

    -m /mnt/models/llama2-7b-exl2/measurement.json \

    -cf /mnt/models/llama2-7b-exl2/4.0bpw/ \

    -b 4.0


Bash


python convert.py \

    -i /mnt/models/llama2-7b-fp16/ \

    -o /mnt/temp/exl2/ \

    -nr \

    -m /mnt/models/llama2-7b-exl2/measurement.json \

    -cf /mnt/models/llama2-7b-exl2/4.5bpw/ \

    -b 4.5


Example 4: Resuming a Job


Bash


python convert.py -o /mnt/temp/exl2/


2. HumanEval Evaluation (eval/humaneval.py) Setup and Execution:


Installation:


Bash


pip install human-eval


Execution:


Bash


python eval/humaneval.py -m <model_dir> -o humaneval_output.json

evaluate-functional-correctness humaneval_output.json


3. MMLU Evaluation (eval/mmlu.py) Setup and Execution:


Installation:


Bash


pip install datasets


Execution:


Bash


python eval/mmlu.py -m <model_dir>


Analysis of the Code Snippets:




convert.py Snippets: These examples demonstrate how to use the convert.py script for EXL2 quantization. They showcase different use cases:


Basic conversion:  Converts an FP16 model to EXL2 with a specified bitrate (-b). It also copies original model files to the output directory (-cf).
Measurement pass:  Runs only the measurement phase of quantization, saving the results to a JSON file (-om). This is useful for reusing measurements for multiple quantization attempts. The -nr flag prevents resuming previous jobs and starts fresh.
Using measurement file:  Skips the measurement pass and uses a pre-calculated measurement file (-m) for faster quantization, especially when trying different bitrates.
Resuming a job: Shows how to resume an interrupted conversion job by simply providing the working directory (-o).




eval/humaneval.py Snippets: These show how to run the HumanEval benchmark:


pip install human-eval:  Installs the necessary human-eval package.
python eval/humaneval.py ...: Executes the HumanEval script, specifying the model directory (-m) and output file (-o).
evaluate-functional-correctness ...:  A separate command (likely from the human-eval package) to evaluate the generated code samples for functional correctness.




eval/mmlu.py Snippets: These show how to run the MMLU benchmark:


pip install datasets: Installs the datasets library, required for MMLU.
python eval/mmlu.py ...: Executes the MMLU script, specifying the model directory (-m).




Key Observations:


Python Scripts: All the main tools (convert.py, humaneval.py, mmlu.py) are Python scripts.
Command-Line Arguments:  The scripts are controlled via command-line arguments, making them flexible and scriptable.
File Paths:  Many arguments involve file paths (-i, -o, -cf, -om, -m, -o in evaluation scripts), indicating that these tools work with models and data stored in files and directories.
Bitrate Control: The -b argument in convert.py is central to controlling the target bitrate for EXL2 quantization.
Evaluation Benchmarks: The documentation provides 
clear examples of how to evaluate EXLlamaV2 models using standard 
benchmarks like HumanEval and MMLU.
Installation Steps:  The documentation includes pip install commands, highlighting the dependencies required for the evaluation scripts.





### **4.3 Advantages of GPTQ: Speed, Compression, NVIDIA GPU Optimization**


*   **Fastest Inference (with ExLlamaV2):** GPTQ, especially with ExLlamaV2, is known for delivering very fast inference speeds compared to other quantization methods, particularly on NVIDIA GPUs. Expect significant speedups compared to running the original full-precision model or using less optimized quantization methods.

*   **Excellent Compression:** GPTQ achieves significant model size reduction, typically compressing models to around 4 bits per weight. This allows you to fit much larger models into limited VRAM. A 7B parameter model might be reduced to around 3.5-4GB in size.

*   **NVIDIA GPU Optimization:** ExLlamaV2 and many GPTQ implementations are highly optimized for NVIDIA GPUs, leveraging CUDA and Tensor Cores for maximum performance. If you have an NVIDIA GPU, GPTQ with ExLlamaV2 is often the top choice for speed.


### **4.4 Trade-offs: Potential Accuracy Loss**


*   **Potential Accuracy Loss:** GPTQ quantization can sometimes lead to a slight accuracy loss compared to the original full-precision model.  The extent of accuracy loss depends on:

    *   **Quantization Level:** 4-bit quantization is more aggressive than 8-bit and might have a slightly larger accuracy impact.

    *   **Model Architecture:** Some models are more robust to quantization than others.

    *   **Task:** The specific task you are performing. Some tasks are more sensitive to quantization than others.

    However, for many use cases, the accuracy loss is minimal and a worthwhile trade-off for the significant performance and memory benefits.  In practice, for many chat-style LLMs, the perceived quality difference after GPTQ quantization is often negligible.


### **4.5 Recipe: Running GPTQ Quantized Models with ExLlamaV2 - Step-by-Step**


Here's a step-by-step recipe to run GPTQ quantized models using AutoGPTQ and Transformers in Python:


1.  **Install Required Libraries:**

    ```

    pip install auto-gptq optimum transformers

    ```

    `auto-gptq` is the library for loading and running GPTQ models. `optimum` is needed for safetensors support (recommended). `transformers` is the Hugging Face library for models and tokenizers.


2.  **Choose a GPTQ Quantized Model:** Find a GPTQ quantized version of your desired model on the Hugging Face Hub. Look for models tagged with "GPTQ" or from users like "TheBloke" who specialize in quantized models.  Example: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ". Check the model card for the correct `model_basename`.


3.  **Load the Model and Tokenizer in Python:** Use the Python code example provided in section 4.7, replacing `"TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"` and `"gptq_model-4bit-128g"` with your chosen model's name and basename.


4.  **Run Inference:** Use the `model.generate()` function as shown in the example to generate text. Adjust generation parameters like `max_new_tokens`, `temperature`, etc., as needed.


5.  **Monitor VRAM Usage:** Use `nvidia-smi` to monitor your GPU VRAM usage. Ensure you are not running out of memory. If you are, try reducing `max_new_tokens` or using a smaller model.


**4.6 Troubleshooting GPTQ Loading: Common Errors and Solutions**


*   **"CUDA out of memory" Error:** Even with a quantized model, you might still run out of VRAM, especially with long context lengths or large generation lengths.

    *   **Solution:** Reduce `max_new_tokens` in `model.generate()`. Try using a smaller model. If using `llama.cpp` with GPTQ (less common, but possible), try offloading more layers to CPU with `-ngl`. Ensure no other VRAM-intensive applications are running.

*   **"ModuleNotFoundError: No module named 'auto_gptq'" or "'optimum'":**

    *   **Solution:** Make sure you have installed `auto-gptq` and `optimum` using `pip install auto-gptq optimum`. Double-check your Python environment.

*   **Model Loading Errors (e.g., "ValueError: ..."):**

    *   **Solution:** Double-check the `model_name_or_path` and `model_basename` are correct and match the Hugging Face Hub model card. Ensure `use_safetensors=True` is set if the model uses safetensors (recommended). `trust_remote_code=True` might be needed for some models - check the model card.


### **4.7 Practical Example with AutoGPTQ and Transformers (Python Code):**


```

# Install AutoGPTQ and optimum (if not already installed)

pip install auto-gptq

pip install optimum  # For safetensors support


from auto_gptq import AutoGPTQForCausalLM

from transformers import AutoTokenizer

import torch

import os


model_name_or_path = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"  # Replace with your model (GPTQ version from Hugging Face Hub)

model_basename = "gptq_model-4bit-128g"  # Replace with your model's base name (check Hugging Face Hub model card)
```

- Load the tokenizer
```
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
```

- Load the quantized model
```
model = AutoGPTQForCausalLM.from_quantized(

    model_name_or_path,

    model_basename=model_basename,

    use_safetensors=True,  # Recommended for faster and safer loading of model weights

    trust_remote_code=True,  # Required for some models

    device="cuda:0",  # Use the first GPU if available, or "cpu" to force CPU usage

    use_triton=False,  # Set to True if you have Triton installed for potentially faster inference (requires Triton installation)

    quantize_config=None,  # Set to None when loading a pre-quantized GPTQ model

)
```

- Example prompt
```
prompt = "Write a short story about a cat who learns to code."
```

- Tokenize the prompt
```
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")  # Move input tensors to GPU if model is on GPU
```

- Generate text
```
with torch.no_grad():  # Disable gradient calculation for inference

    generation_output = model.generate(

        **inputs,  # **inputs unpacks the dictionary returned by the tokenizer

        max_new_tokens=100,  # Maximum number of tokens to generate

        do_sample=True,      # Enable sampling for more diverse outputs

        top_k=50,           # Top-k sampling parameter

        top_p=0.95,          # Top-p (nucleus) sampling parameter

        temperature=0.7,     # Sampling temperature (higher = more random)

    )
```

- Decode the output
```
generated_text = tokenizer.decode(generation_output[0])

print(generated_text)
```

- Print model size
```
print(f"Number of parameters: {model.num_parameters()}")

model_size_mb = os.path.getsize(os.path.join(model_name_or_path, model_basename + ".safetensors")) / (1024**2)  # Assuming safetensors

print(f"Quantized model size: {model_size_mb:.2f} MB")  # Or GB if larger
```

## 5: GGML/GGUF and llama.cpp - CPU and Cross-Platform Efficiency

### 5.1 Introduction to GGML/GGUF Model Format

GGML (Go Get Machine Learning) / GGUF (successor to GGML) is a model format specifically designed for efficient inference, particularly on CPUs. It is closely associated with the llama.cpp library, which is a highly optimized C++ library for running LLMs. GGUF is designed for cross-platform compatibility and efficient CPU inference, making it excellent for systems without powerful GPUs or when you want to utilize the CPU for inference.

### 5.2 llama.cpp Library: CPU-Optimized Inference

llama.cpp is a powerful C++ library that provides:

    Inference Engine: A highly optimized inference engine for GGML/GGUF models, written in C++ for speed and efficiency, especially on CPUs.
    Quantization Tools: Tools to convert models to GGML/GGUF format and apply different quantization schemes (though often pre-quantized GGUF models are downloaded).
    Example Applications: Includes example applications like main (for command-line inference) and server (for a simple web server).

llama.cpp is renowned for its ability to run surprisingly large LLMs efficiently on CPUs, even on modest hardware like laptops.

### 5.3 Versatility: CPU and GPU Support, Various Quantization Levels

    CPU Optimization: GGML/GGUF and llama.cpp are primarily CPU-focused and highly optimized for CPU inference. They leverage CPU resources effectively, making them ideal for systems without powerful GPUs.
    GPU Support: While primarily CPU-focused, llama.cpp also supports GPU acceleration via:
        CUDA (NVIDIA): For NVIDIA GPUs.
        ROCm (AMD): For AMD GPUs.
        Metal (Apple Silicon): For macOS with Apple Silicon GPUs.
        This allows you to offload some computation to the GPU if available, further speeding up inference.
    Various Quantization Levels: GGML/GGUF supports a wide range of quantization levels, offering flexibility in balancing model size, speed, and accuracy. Examples include:
        Q4_0, Q4_K_M, Q5_K_S, etc.
        Q indicates quantization, the number (e.g., 4, 5) roughly represents bits per weight.
        Suffixes like K and M denote different quantization schemes (K-quants often offer better accuracy for similar size, M-quants might be smaller).
        Lower quantization levels generally mean smaller model size and faster inference but potentially lower accuracy. Refer to llama.cpp documentation for detailed explanations of each quantization type.

### 5.4 Cross-Platform Compatibility (Linux, macOS, Windows)

GGUF models and llama.cpp are highly cross-platform, working well on Linux, macOS, and Windows. This makes them a versatile choice for development and deployment across different operating systems.

5.5 Recipe: Running GGUF Models with llama.cpp - Step-by-Step

Here's a step-by-step recipe to run GGUF quantized models using llama.cpp:

    Download llama.cpp: Clone the llama.cpp repository from GitHub:

```

git clone https://github.com/ggerganov/llama.cpp

    cd llama.cpp
```

    Build llama.cpp: Compile llama.cpp from source. The build process varies slightly depending on whether you want CPU-only or GPU acceleration. See section 5.7 for build commands. Ensure you have a C++ compiler (like g++) and CMake installed.

    Download a GGUF Model: Download a pre-quantized GGUF model file. Hugging Face Hub is a good source. Look for models in GGUF format, often provided by "TheBloke". Example: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF". Create a models directory inside the llama.cpp directory and download the .gguf file there using wget or curl.

    Run Inference with ./main: Use the ./main executable (built in step 2) to run inference. See section 5.7 for example commands and parameter explanations.

    Adjust Parameters: Experiment with parameters like -n (max tokens), -p (prompt), -t (threads), and -ngl (GPU layers) to optimize performance and resource usage. Monitor CPU and GPU usage using system monitoring tools.

### 5.6 Troubleshooting llama.cpp Build Issues: Common Compiler Errors and Dependencies

    "make: command not found" or "g++: command not found":
        Solution: Ensure you have a C++ compiler (like g++) and make installed. On Debian/Ubuntu-based systems, install with: sudo apt-get install build-essential cmake. On Fedora/CentOS/RHEL: sudo yum groupinstall "Development Tools" cmake.
    "CMake Error: CMake was unable to find a build program...":
        Solution: Ensure CMake is installed. On Debian/Ubuntu: sudo apt-get install cmake. On Fedora/CentOS/RHEL: sudo yum install cmake.
    CUDA/ROCm Build Errors (if using GPU acceleration):
        Solution: Double-check that CUDA or ROCm is correctly installed and configured on your system. Verify that nvcc --version (for CUDA) or hipcc --version (for ROCm) works. Ensure the CUDA_DIR or HIP_DIR paths in the make command are correct. Consult the llama.cpp README for detailed GPU build instructions and troubleshooting.
    "Illegal instruction" or crashes at runtime:
        Solution: This can sometimes be due to CPU architecture incompatibilities or issues with specific quantization levels. Try building llama.cpp with different CPU optimization flags (check llama.cpp documentation) or try a different GGUF quantization level.

### 5.7 Practical Example: Building llama.cpp and Running Inference (Bash Commands)



- Download llama.cpp:
```
git clone https://github.com/ggerganov/llama.cpp

cd llama.cpp
```

Build llama.cpp: (ensure you have a C++ compiler and CMake installed)

- CPU only (Default CPU build with OpenBLAS)
```
make
```

- If you have an NVIDIA GPU, add the following flags (adjust for your CUDA version):
```
make LLAMA_CUDA=1 CUDA_DIR=/usr/local/cuda  # For NVIDIA GPUs using cuBLAS. Ensure CUDA is installed. CUDA_DIR typically is /usr/local/cuda or /usr/cuda. Check 'nvcc --version' to confirm CUDA installation.
```

- (Optional) AMD GPU (ROCm):
```
make LLAMA_HIP=1 HIP_DIR=/opt/rocm  # For AMD GPUs using ROCm. Ensure ROCm is installed. HIP_DIR is often /opt/rocm. Check 'hipcc --version' to confirm ROCm installation.
```

- Optional) macOS Metal:
```
make LLAMA_METAL=1  # For macOS with Apple Silicon GPUs using Metal.
```

- Download a GGUF model: (e.g., TinyLlama-1.1B-Chat-v1.0-Q4_K_M)
```
mkdir -p models

wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

- Run inference:
```
./main -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -n 100 -p "Write a short story about a robot..." -t 8 -ngl 20
```

Explanation of main command parameters:

    - ./main: Executes the main` binary (the llama.cpp inference program).
    -m models/tinyllama-1.1b-chat-v1.0-Q4_K_M.gguf: Specifies the path to the GGUF model file.
    -n 100: Sets the maximum number of tokens to generate (100 in this example).
    -p "Write a short story about a robot...": Provides the prompt for text generation.
    -t 8: Number of threads/



threads to use for CPU inference. Adjust this based on your CPU core count. Start with the number of physical cores your CPU has.

*   `-ngl 20`:  `-ngl` (number of GPU layers) is an *optional* parameter for GPU acceleration. It specifies how many layers of the neural network to offload to the GPU.  If you have a GPU and built llama.cpp with GPU support (CUDA, ROCm, or Metal), you can use `-ngl` to offload computation to the GPU, potentially speeding up inference.  The optimal value for `-ngl` depends on your GPU VRAM and the model size. Start with a value like `20` and experiment.  A higher value offloads more layers to the GPU, which can be faster if you have enough VRAM. If you don't have a GPU or don't want to use it, omit the `-ngl` parameter.


---


## **6: Bitsandbytes - Easy Quantization in Transformers**


### **6.1 Introduction to Bitsandbytes Library**


Bitsandbytes is a Python library that simplifies the process of using quantization, especially 8-bit and 4-bit quantization, within the Hugging Face Transformers ecosystem and PyTorch. It provides custom PyTorch optimizers and functions that allow you to load and train models in lower precision formats with minimal code changes. For inference, bitsandbytes primarily focuses on 4-bit quantization for *loading* models, enabling you to fit larger models into GPU memory.


### **6.2 Ease of Integration with Hugging Face Transformers**


One of the key advantages of bitsandbytes is its seamless integration with the popular Hugging Face Transformers library. You can load pre-trained models from the Hugging Face Hub in 4-bit or 8-bit quantized format with just a few extra lines of code when using `AutoModelForCausalLM.from_pretrained()` or similar functions. This makes it very easy to experiment with quantization and incorporate it into existing Transformers-based workflows.


### **6.3 Support for 4-bit and 8-bit Quantization**


Bitsandbytes primarily focuses on:


*   **4-bit Quantization (FP4, NF4):** Bitsandbytes offers different 4-bit quantization types, including FP4 (4-bit floating point) and NF4 (NormalFloat4), which is often recommended for better performance in LLMs. 4-bit quantization provides the most aggressive memory reduction, allowing you to load very large models even on GPUs with limited VRAM.

*   **8-bit Quantization (INT8):** Bitsandbytes also supports 8-bit quantization (INT8). While 8-bit quantization provides less memory saving than 4-bit, it can still offer a good balance of memory reduction and minimal accuracy impact. 8-bit quantization can also be useful for accelerating matrix multiplications, especially on CPUs and GPUs with INT8 acceleration capabilities.


**6.4 Mixed-Precision Strategies**


Bitsandbytes often employs mixed-precision strategies under the hood. This means that while the model weights are stored in a lower precision format (like 4-bit), computations might be performed in a higher precision (like FP16 or BF16) for better numerical stability and accuracy. This is particularly relevant for 4-bit quantization, where computations in very low precision alone might lead to accuracy degradation. Bitsandbytes handles these mixed-precision details automatically, making it easy for users.


### **6.5 Use Cases: Quick Experimentation, Python Workflows, Memory-Efficient Training**


Bitsandbytes is well-suited for several use cases:


*   **Quick Experimentation with Quantization:** Its ease of use makes it ideal for quickly trying out quantization and seeing its impact on memory usage and performance in Python-based LLM projects.

*   **Python-Centric Workflows:** If your LLM workflow is primarily in Python and uses Hugging Face Transformers, bitsandbytes provides a natural and easy way to integrate quantization.

*   **Memory-Efficient Training (Advanced):** While this guide focuses on inference, bitsandbytes also supports memory-efficient training techniques like 8-bit optimizers (e.g., `bnb.optim.Adam8bit`). This is more advanced and beyond the scope of this inference-focused guide, but worth noting for users interested in training.


### **6.6 Recipe: Loading a 4-bit Quantized Model with Bitsandbytes - Step-by-Step**


Here's a recipe to load a model in 4-bit using bitsandbytes with Hugging Face Transformers:


1.  **Install Bitsandbytes and Transformers:**

    ```

    pip install bitsandbytes transformers

    ```

    Ensure you have both libraries installed in your Python environment.


2.  **Choose a Model ID:** Select a model from the Hugging Face Hub that you want to load in 4-bit.  You can start with a smaller model for testing, like `facebook/opt-350m`.


3.  **Load Model with `load_in_4bit=True`:** Use `AutoModelForCausalLM.from_pretrained()` (or similar Transformers loading functions) and set the `load_in_4bit=True` argument.  Optionally, you can also set `torch_dtype=torch.float16` for mixed-precision (recommended for 4-bit). Set `device_map='auto'` to let Transformers automatically manage device placement (GPU if available, CPU if not).


4.  **Run Inference:** Use the loaded model for inference as you would with a standard Transformers model (e.g., using `model.generate()`).


5.  **Monitor Memory Usage:** Check GPU VRAM usage with `nvidia-smi` to confirm that 4-bit loading has reduced memory footprint.


### **6.7 Practical Example: Loading a 4-bit Quantized Model with Transformers (Python Code)**


```
Install bitsandbytes and transformers (if not already installed)

pip install bitsandbytes

pip install transformers


from transformers import AutoModelForCausalLM, AutoTokenizer

import torch


model_id = "facebook/opt-350m"  # Replace with your model ID from Hugging Face Hub
```

- Load tokenizer (standard transformers way)
```
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

- Load model in 4-bit using bitsandbytes
```
model_4bit = AutoModelForCausalLM.from_pretrained(

    model_id,

    device_map='auto',

    load_in_4bit=True,

    torch_dtype=torch.float16  # Load in 4-bit with float16 for mixed-precision operations

)
```

- Example prompt
```
prompt = "Write a poem about the Linux operating system."
```

- Tokenize the prompt
```
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")  # Move inputs to GPU
```

- Generate text
```
with torch.no_grad():

    outputs = model_4bit.generate(**inputs, max_new_tokens=50)
```

- Decode and print output
```
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
```

## 7: Model Selection - Choosing Efficient LLM Architectures

### 7.1 Importance of Model Architecture for Efficiency

Beyond quantization, the choice of LLM architecture itself plays a crucial role in determining inference efficiency. Different model architectures have varying levels of computational complexity and memory requirements. Selecting a more efficient architecture can be as impactful as, or even more impactful than, post-hoc optimization techniques like quantization.

### 7.2 Choosing Smaller Models vs. Larger Models

The most straightforward way to reduce resource consumption is to choose smaller LLMs. Model size is often measured by the number of parameters (e.g., 7B, 13B, 70B, 175B).

    Smaller Models (e.g., < 7B parameters):
        Advantages: Significantly lower VRAM and system RAM requirements. Faster inference speed. Can run on CPUs or low-end GPUs.
        Disadvantages: Generally lower overall performance compared to larger models, especially on complex tasks. Might produce less coherent or less nuanced text.
        Examples: TinyLlama-1.1B, Phi-2 (2.7B), smaller OPT or GPT models.

    Larger Models (e.g., 7B - 70B+ parameters):
        Advantages: Higher potential performance, better coherence, more nuanced text generation, better at complex tasks.
        Disadvantages: Higher VRAM and system RAM requirements. Slower inference speed. Might require GPUs for acceptable performance.
        Examples: Mistral 7B, Llama 2 (7B, 13B, 70B), larger OPT or GPT models.

The "best" model size depends entirely on your specific use case, resource constraints, and desired output quality. For resource-limited environments, starting with smaller models and then scaling up as needed is a good strategy.

### 7.3 Distilled/Optimized Models (e.g., MobileBERT, DistilBERT, Smaller LLM Variants)

"Distillation" is a technique where a smaller "student" model is trained to mimic the behavior of a larger "teacher" model. This can result in smaller models that retain a surprisingly good level of performance compared to their larger counterparts.

    Distilled Models:
        Examples (for NLP tasks, though less common for generative LLMs): DistilBERT (distilled version of BERT), MobileBERT.
        Advantages: Smaller size, faster inference, often retain a significant portion of the teacher model's performance.
        Disadvantages: Might still not be as performant as the original large model, especially for very complex tasks.

    Optimized Model Variants: Some model families offer specifically optimized "small" or "efficient" variants. For example, within the Llama 2 family, the 7B and 13B models are considerably smaller and more resource-friendly than the 70B model, while still offering strong performance.

### 7.4 Tokenizer Efficiency (SentencePiece, Tiktoken, BPE)

The tokenizer, which converts text into numerical tokens that the model processes, also impacts efficiency. Different tokenizers have varying levels of efficiency in terms of:

    Vocabulary Size: Smaller vocabularies can sometimes lead to slightly smaller model sizes and faster tokenization/detokenization.
    Token Length: Some tokenizers might represent words or sub-word units more efficiently (fewer tokens per word).
    Tokenization Speed: The speed of the tokenization and detokenization process itself can also contribute to overall inference latency, although this is usually less of a bottleneck than model computation.

Common tokenizer types include:

    SentencePiece: Used by many models, including some Google models and multilingual models.
    Tiktoken: OpenAI's tokenizer, known for its efficiency and used by GPT models.
    Byte-Pair Encoding (BPE): A widely used subword tokenization algorithm.

While tokenizer choice is less of a primary optimization target compared to model size and quantization, being aware of tokenizer efficiency is helpful.

### 7.5 Specific Model Recommendations for Resource-Constrained Systems

For resource-constrained systems, consider these model families and specific models:

    TinyLlama Family: TinyLlama-1.1B is an excellent starting point for very resource-limited environments. It's surprisingly capable for its size and can run efficiently even on CPUs.
    Phi Family (Microsoft Phi-1.5, Phi-2): Phi models are designed for efficiency and achieve strong performance for their size. Phi-2 (2.7B parameters) is particularly noteworthy for its capabilities relative to its size.
    Mistral 7B: Mistral 7B is a highly performant 7B parameter model that often outperforms larger models in some benchmarks. It's a good balance of size and capability.
    Smaller Llama 2 Variants (7B, 13B): Llama 2 7B and 13B models are more resource-friendly than the 70B version while still offering strong performance.
    OPT (Open Pretrained Transformer) Family: Smaller OPT models (e.g., OPT-350m, OPT-1.3b) can be useful for experimentation and resource-constrained settings.

### 7.6 Using Hugging Face Hub Filters to Find Efficient Models

The Hugging Face Hub provides filters to help you find models based on size and other criteria:

    Model Size Filter: On the Hugging Face Hub website, you can often filter models by parameter size (e.g., "<1B", "<10B", etc.).
    Task Filters: Filter by the task you are interested in (e.g., "text-generation", "conversational").
    Tags: Look for tags like "quantized", "distilled", "efficient" to find optimized models.
    Sorting by Downloads/Likes: Models with more downloads or "likes" are often popular and well-regarded in the community.

### 7.7 Model Selection Criteria Beyond Size: Task Suitability and Architecture

When selecting a model, consider not just size but also:

    Task Suitability: Is the model trained for the type of task you need (e.g., chat, code generation, general text generation)? Some models are specialized.
    Architecture Innovations: Some newer architectures might be inherently more efficient than older ones. Keep an eye on research papers and model releases that highlight efficiency improvements.
    Community Benchmarks: Look for community benchmarks and evaluations that compare the performance of different models on tasks relevant to you.

Choosing the right model architecture is a fundamental optimization step. By prioritizing smaller, distilled, or efficient model variants, you can significantly reduce resource requirements and improve inference speed, often without drastically sacrificing output quality, especially for many common LLM use cases.

## 8: Offloading to System RAM and NVMe - Expanding Memory Capacity

### 8.1 Concept of Model Offloading

Model offloading is a technique to overcome VRAM limitations by moving parts of the LLM from GPU VRAM to system RAM or even slower storage like NVMe SSDs. When your model is too large to fit entirely into VRAM, offloading becomes necessary to run inference.

### 8.2 VRAM Overflow and Necessity of Offloading

As discussed in Chapter 2, VRAM is a critical resource for GPU-accelerated LLM inference. If the model size, along with activations and KV cache, exceeds the available VRAM, you will encounter "out-of-memory" errors. Offloading provides a way to handle this VRAM overflow by temporarily moving some model layers or data to system RAM or NVMe during inference.

#### 8.3 Offloading to System RAM: Speed Trade-off

Offloading to system RAM is the most common type of offloading.

    Mechanism: When offloading to system RAM, some layers of the neural network (or parts of the model's state) are moved from VRAM to system RAM. When these offloaded layers are needed for computation during inference, they are transferred back to VRAM, computations are performed, and then results might be moved back to system RAM if necessary.
    Speed Trade-off: Accessing system RAM is significantly slower than accessing VRAM. Data needs to be transferred over the PCIe bus between the GPU and system RAM. This data transfer introduces latency and reduces inference speed. Inference with system RAM offloading will always be slower than running the entire model in VRAM. However, it allows you to run models that would otherwise be impossible to load due to VRAM constraints.

### 8.4 Offloading to NVMe SSD: Better Performance than HDD, Still Slower than VRAM

For even larger models that might exceed both VRAM and system RAM capacity, offloading to NVMe SSD (Non-Volatile Memory express Solid State Drive) is an option.

    Mechanism: Similar to system RAM offloading, but data is moved to NVMe SSD storage. NVMe SSDs are much faster than traditional HDDs (Hard Disk Drives) but still significantly slower than system RAM and VRAM.
    Performance: Offloading to NVMe is slower than system RAM offloading but can still be faster than HDD offloading (which is generally not recommended for LLM inference due to extreme slowness). NVMe offloading is a last resort for very large models on systems with very limited VRAM and system RAM.

### 8.5 Recipe: Model Offloading with accelerate - Step-by-Step

The accelerate library from Hugging Face simplifies model offloading in PyTorch-based Transformers models. Here's a recipe:

    Install accelerate:

```

pip install accelerate
```

Load Model with Device Mapping: When loading your model using AutoModelForCausalLM.from_pretrained(), use the device_map='auto' argument. accelerate will automatically determine the best device placement based on available resources (GPU and system RAM).

```

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your_model_id", device_map='auto')
```

(Optional) Control Device Mapping with infer_auto_device_map: For more fine-grained control, you can use infer_auto_device_map to specify memory limits for each device (GPU and CPU). This allows you to control how much VRAM to use and how much system RAM to allow for offloading.

```

from accelerate import infer_auto_device_map

from transformers import AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained("your_model_id") # Load model first

device_map = infer_auto_device_map(model, max_memory={0: "10GB", "cpu": "32GB"}) # Limit GPU 0 to 10GB VRAM, allow up to 32GB system RAM

model = AutoModelForCausalLM.from_pretrained("your_model_id", device_map=device_map) # Reload with device_map
```

In this example, we limit GPU 0 to using a maximum of 10GB of VRAM and allow offloading to system RAM up to 32GB. Adjust these values based on your system's resources.

Dispatch Model: After loading with device_map, use dispatch_model(model) from accelerate to actually move the model parts to the specified devices.

```

from accelerate import dispatch_model

    model = dispatch_model(model)
```

    Run Inference: Proceed with inference as usual using model.generate(). accelerate will handle the data movement between VRAM and system RAM (or NVMe if configured) behind the scenes.

    Monitor Performance: Be aware that offloading will slow down inference. Monitor inference speed and adjust offloading settings (e.g., memory limits in infer_auto_device_map) to find a balance between memory usage and performance.

#### 8.6 Monitoring I/O Performance with iostat

When offloading to system RAM or especially NVMe, I/O (Input/Output) performance becomes more important. You can use the iostat command on Linux to monitor disk I/O statistics.

```

iostat -x 1
```

This command will show extended I/O statistics every 1 second. Look at columns like %util (disk utilization), await (average wait time for I/O requests), rMB/s (read MB per second), and wMB/s (write MB per second) to understand disk I/O activity during inference with offloading. High disk utilization and long wait times can indicate I/O bottlenecks.

#### 8.7 Using accelerate Library for Model Offloading (Conceptual Example)

```

Install accelerate (if not already installed)

pip install accelerate


from accelerate import dispatch_model, infer_auto_device_map

from transformers import AutoModelForCausalLM


model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"  # Replace with your model ID

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')  # 'auto' automatically decides device mapping
```

- OR, for more control:
```
device_map = infer_auto_device_map(model, max_memory={0: "10GB", "cpu": "32GB"})  # Limit GPU 0 memory to 10GB, allow up to 32GB system RAM

model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map)


model = dispatch_model(model)  # Dispatch model to devices according to device_map

print(model.device_map)  # Print the device map to see layer placement
```

Model offloading is a powerful technique for running very large LLMs that exceed VRAM capacity. However, be mindful of the performance trade-offs, especially when offloading to system RAM or NVMe. Optimize your offloading strategy and monitor performance to find the best balance for your specific hardware and model.

## 9: Memory Mapping (mmap) for Efficient Model Loading

### 9.1 Concept of Memory Mapping (mmap)

Memory mapping, often referred to as mmap (memory map), is a powerful operating system feature that can significantly enhance the efficiency of loading large files, such as LLM model weights, into memory. It provides a way to directly map a file on disk to a process's virtual memory space, eliminating the need for traditional read/write system calls for accessing file data.

### 9.2 Benefits of mmap for LLM Loading: Speed, Memory Efficiency

For loading large LLM models, mmap offers several key advantages:

    Speed:
        Faster Startup Time: With mmap, the initial "loading" of the model becomes very fast. Instead of physically copying the entire multi-gigabyte model file into RAM at startup, mmap primarily sets up the virtual memory mapping. The actual loading of data from disk into RAM is deferred and happens lazily as the model is used during inference. This significantly reduces the initial startup time of applications using large models.
        Potentially Faster Access: In some cases, accessing data through mmap can be faster than traditional file I/O. The operating system's memory management system is often highly optimized for demand paging and can efficiently manage the transfer of data between disk and RAM.

    Memory Efficiency:
        Reduced Memory Footprint: mmap can lead to a lower initial memory footprint. Only the actively used parts of the model are loaded into RAM. If your inference workload only accesses a subset of the model parameters at any given time, mmap can prevent the entire model from being loaded into RAM simultaneously.
        Shared Memory Potential: If multiple processes need to access the same model file, mmap can enable efficient sharing of the model data in RAM. Multiple processes can map the same file into their virtual address spaces. The operating system can then share the physical RAM pages containing the model data between these processes, reducing overall memory consumption when running multiple instances of an LLM-based application.

### 9.3 How mmap Works with Model Files (e.g., .safetensors, .gguf)

Modern model file formats like .safetensors and .gguf are often designed to be efficiently used with mmap. These formats typically store model weights in a contiguous layout on disk, which is well-suited for memory mapping. Libraries that load these formats (like safetensors Python library or llama.cpp for GGUF) often internally leverage mmap to load model weights.

When you load a .safetensors or .gguf model using a library that utilizes mmap, the library essentially:

    Opens the model file.
    Uses the mmap system call to map the file into the process's virtual memory.
    Returns a pointer or data structure that allows you to access the model weights as if they were directly in memory.

Behind the scenes, the operating system manages the actual loading of data from disk into RAM as your application accesses the model weights through this memory mapping.

### 9.4 Practical Considerations:

    File System Caching: Operating systems aggressively use file system caching. Even without explicit mmap, if you repeatedly access the same parts of a model file through traditional file I/O, the operating system's cache might keep frequently accessed data in RAM. However, mmap provides a more direct and controlled way to leverage this caching behavior and can be more efficient for large files.

    Shared Memory: The shared memory aspect of mmap is particularly beneficial in scenarios where you are running multiple LLM inference processes concurrently, such as in a server environment. By using mmap, you can potentially reduce the total RAM usage across all processes, as they can share the underlying model data in memory.

#### 9.5 Python Libraries Leveraging mmap:

    safetensors library: The safetensors library in Python is explicitly designed to use mmap for efficient loading of .safetensors model files. When you load a .safetensors file using safetensors.torch.load, it internally uses mmap to map the file into memory. This is a key reason why .safetensors is often recommended for fast and memory-efficient model loading in Python-based LLM workflows.

    llama.cpp: The llama.cpp library, used for GGUF models, also leverages mmap for efficient model loading. This contributes to the fast startup times and memory efficiency of llama.cpp, especially when running on CPUs.

In summary, memory mapping (mmap) is a valuable technique for optimizing LLM loading, especially for large models on resource-constrained systems. It offers faster startup times, potentially faster access, and improved memory efficiency by leveraging demand paging and shared memory capabilities of the operating system. Libraries like safetensors and llama.cpp effectively utilize mmap to provide these benefits when working with modern LLM model formats.

## 10: Compilation and Graph Optimization - Speeding Up Inference

### 10.1 Just-In-Time (JIT) Compilation for LLMs

Just-In-Time (JIT) compilation involves compiling parts of the model's computation graph into optimized machine code at runtime, specifically tailored to the hardware and input characteristics. Instead of interpreting the model's operations step-by-step, JIT compilation generates native machine code for critical parts of the computation, leading to substantial performance gains.

### 10.2 Graph Optimization Techniques

Compilation often goes hand-in-hand with graph optimization. These techniques aim to restructure the model's computational graph to improve efficiency before or during compilation. Common graph optimization techniques include:

    Operator Fusion: Combining multiple small operations (operators) into a single, larger, more efficient operator. For example, fusing a sequence of element-wise operations (like addition, multiplication, ReLU) into a single fused kernel can reduce overhead and improve data locality.
    Kernel Optimization: Replacing generic operator implementations with highly optimized kernels specifically written for the target hardware architecture (e.g., using highly optimized CUDA kernels for NVIDIA GPUs, or optimized kernels for specific CPU architectures).
    Memory Layout Optimization: Rearranging data in memory to improve memory access patterns and cache utilization, reducing memory bandwidth bottlenecks.
    Pruning and Sparsity Exploitation: Removing or zeroing out less important connections (weights) in the model (pruning) and then exploiting this sparsity in computation to reduce the number of operations.

### 10.3 Libraries for Compilation and Optimization

Several libraries and frameworks provide tools for compilation and graph optimization of LLMs:

    TorchScript (PyTorch): TorchScript is PyTorch's JIT compiler and serialization format. You can convert PyTorch models to TorchScript and then run them using the TorchScript interpreter or compile them ahead-of-time. TorchScript can enable graph optimizations and improve inference speed, especially for CPU inference.
    ONNX Runtime (Microsoft): ONNX Runtime is a cross-platform inference engine that supports ONNX (Open Neural Network Exchange) models. You can export models from various frameworks (including PyTorch and TensorFlow) to ONNX and then run them using ONNX Runtime. ONNX Runtime performs graph optimizations and provides optimized execution backends for various hardware platforms (CPUs, GPUs).
    TensorRT (NVIDIA): TensorRT is a high-performance inference SDK from NVIDIA specifically designed for NVIDIA GPUs. It takes trained models (from frameworks like TensorFlow, PyTorch, ONNX) and applies graph optimizations, layer fusion, and kernel auto-tuning to maximize inference throughput and minimize latency on NVIDIA GPUs. TensorRT is particularly effective for achieving very low latency inference.
    DeepSpeed (Microsoft): DeepSpeed, while primarily known for distributed training, also offers inference optimization features, including DeepSpeed-Inference. DeepSpeed-Inference provides kernel optimizations, quantization, and efficient inference kernels, especially for large models and sequence generation tasks.

### 10.4 Trade-offs: Compilation Time vs. Inference Speedup, Hardware Compatibility

    Compilation Time: Compilation, especially with advanced techniques like TensorRT, can sometimes take a significant amount of time upfront. This compilation step is a one-time cost. The trade-off is that this initial compilation time is often offset by much faster inference speeds afterwards.
    Inference Speedup: The speedup achieved through compilation and optimization can be substantial, often ranging from 2x to 10x or even more compared to unoptimized execution, depending on the model, hardware, and optimization techniques used.
    Hardware Compatibility: Some compilation and optimization techniques are highly hardware-specific. For example, TensorRT is primarily designed for NVIDIA GPUs. ONNX Runtime aims for broader hardware compatibility but might offer different levels of optimization on different platforms. TorchScript is generally more portable but might offer less aggressive optimization than specialized tools like TensorRT.

### 10.5 Conceptual Examples and Tools



### 10.6 TorchScript Example (PyTorch Code)

```

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "facebook/opt-350m"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

model.eval()  # Set model to evaluation mode
```

- Example input
```
example_input = tokenizer("Hello, world!", return_tensors="pt")
```

- race the model to create a TorchScript module
```
scripted_model = torch.jit.trace(model, example_input.input_ids)
```

- Save the TorchScript model
```
torch.jit.save(scripted_model, "opt_350m_scripted.pt")
```

- Load and run the TorchScript model (for inference)
```
loaded_scripted_model = torch.jit.load("opt_350m_scripted.pt")
```

with torch.no_grad():
```
    output = loaded_scripted_model(**example_input) # Use the scripted model for inference

    # ... generate text using the output ...
```

    TensorRT (Conceptual - Requires NVIDIA GPU and TensorRT Installation):

TensorRT typically involves a more complex workflow, often using the TensorRT Python API or command-line tools to convert a model (e.g., from ONNX) to a TensorRT engine. The process usually involves:

    Exporting the model to ONNX format.
    Using the TensorRT trtexec command-line tool or Python API to build a TensorRT engine from the ONNX model, specifying target hardware, precision (e.g., FP16, INT8), and optimization settings.
    Loading the TensorRT engine and using the TensorRT inference API for fast inference.

Choosing the right compilation and optimization approach depends on your specific needs, target hardware, and acceptable trade-offs between compilation time and inference speed. For maximum performance on NVIDIA GPUs, TensorRT is often the top choice. For broader hardware compatibility and easier integration with PyTorch, TorchScript and ONNX Runtime are valuable options. DeepSpeed-Inference is particularly relevant for very large models and sequence generation tasks.

## 11: Hardware Acceleration - Leveraging GPUs and Specialized Hardware

### 11.1 Benefits of GPUs for LLM Inference

GPUs (Graphics Processing Units) are massively parallel processors originally designed for graphics rendering but are exceptionally well-suited for the matrix multiplications and other linear algebra operations that are at the heart of LLM computations.

    Parallel Processing: GPUs have thousands of cores that can perform computations in parallel. This massive parallelism aligns perfectly with the parallel nature of matrix operations in neural networks.
    High Memory Bandwidth: GPUs have significantly higher memory bandwidth compared to CPUs. This is crucial for feeding the GPU cores with the large amounts of data (model weights, activations) required for LLM inference.
    Specialized Instructions: Modern GPUs often include specialized hardware units and instructions (e.g., Tensor Cores on NVIDIA GPUs) that are optimized for deep learning operations, further accelerating matrix multiplications and convolutions.

### 11.2 NVIDIA GPUs and CUDA

NVIDIA GPUs, especially their high-end consumer and professional lines (GeForce RTX, NVIDIA RTX, Tesla/A-series, H-series), are the most widely used GPUs for deep learning and LLM inference. NVIDIA's CUDA (Compute Unified Device Architecture) is a parallel computing platform and API that allows developers to program NVIDIA GPUs for general-purpose computations, including deep learning. Libraries like PyTorch, TensorFlow, TensorRT, and ExLlamaV2 are heavily optimized for CUDA and NVIDIA GPUs.

### 11.3 AMD GPUs and ROCm

AMD GPUs, particularly their Radeon and Radeon Pro series, are also increasingly used for deep learning. ROCm (Radeon Open Compute platform) is AMD's open-source alternative to CUDA. ROCm provides a software stack for GPU-accelerated computing on AMD GPUs. While the ecosystem and library support for ROCm might be slightly less mature than CUDA, ROCm is actively developing and improving, and libraries like llama.cpp and some PyTorch/TensorFlow components support ROCm.

### 11.4 Apple Silicon GPUs and Metal

Apple Silicon Macs (M1, M2, M3 chips) integrate powerful GPUs directly into the system-on-a-chip (SoC). Apple's Metal framework provides a low-level API for accessing and programming these GPUs for compute tasks. Libraries like llama.cpp and some PyTorch backends support Metal for GPU acceleration on Apple Silicon Macs. Metal offers excellent performance and efficiency on Apple's hardware.

### 11.5 CPUs with Vector Extensions (AVX-512, AVX2)

While GPUs are generally much faster for LLM inference, modern CPUs also have capabilities for acceleration, particularly through vector extensions like AVX-512 and AVX2 (Advanced Vector Extensions). These extensions allow CPUs to perform Single Instruction, Multiple Data (SIMD) operations, processing multiple data elements simultaneously with a single instruction. Libraries like llama.cpp and optimized CPU inference backends in frameworks like PyTorch and ONNX Runtime leverage AVX-512 and AVX2 to improve CPU inference performance. AVX-512, when available (primarily on high-end Intel and some AMD CPUs), can offer significant speedups for CPU inference.

### 11.6 Specialized AI Accelerators (Conceptual Overview)

Beyond GPUs and CPUs, there are specialized AI accelerators designed specifically for deep learning workloads. These include:

    TPUs (Tensor Processing Units - Google): TPUs are custom ASICs (Application-Specific Integrated Circuits) developed by Google specifically for accelerating machine learning workloads, particularly TensorFlow models. TPUs are highly optimized for matrix multiplications and offer very high performance for certain types of deep learning models. TPUs are primarily available through Google Cloud and Google Colab.
    Inferentia (AWS): AWS Inferentia is another example of a custom AI accelerator, developed by Amazon Web Services. Inferentia is designed for cost-effective and high-performance inference of deep learning models in the cloud.

### 11.7 Choosing the Right Hardware for Your Budget and Performance Needs - Specific Recommendations

Selecting the appropriate hardware for running LLMs depends on your budget, performance requirements, and use case. Here are more specific hardware recommendations:

    Budget-Constrained/Entry Level (For Experimentation and Smaller Models):


*   **CPU-Only:**  For very small models (e.g., TinyLlama-1.1B) and basic experimentation, a modern multi-core CPU (Intel Core i5/i7 or AMD Ryzen 5/7 or equivalent) with at least 16GB of RAM can be sufficient, especially when using llama.cpp and GGUF models. Look for CPUs with AVX2 or AVX-512 support if possible for better performance.  Integrated graphics are fine for CPU-only inference. This is the most budget-friendly option.

*   **Used/Older NVIDIA GPUs:**  Consider used or older generation NVIDIA GPUs like GeForce GTX 1080 Ti, RTX 2070, or Quadro P series cards. These can be acquired at lower prices and still offer decent VRAM (8GB-11GB or more) for running moderately sized quantized models. Ensure they are CUDA-compatible.

*   **Mid-Range (Good Balance of Performance and Cost):**

    *   **NVIDIA GeForce RTX 3060 (12GB VRAM):**  Excellent value for money. 12GB VRAM allows running larger quantized models. Good CUDA performance.

    *   **NVIDIA GeForce RTX 4060 Ti (16GB VRAM):**  More VRAM than RTX 3060, offering even better capacity for larger models.

    *   **AMD Radeon RX 6700 XT or RX 6800 (12GB-16GB VRAM):**  Competitive performance and VRAM at a good price point.  ROCm support is improving, making them viable for LLMs, though NVIDIA still has a stronger ecosystem.

*   **High-End Consumer (For Larger Models and Faster Inference):**

    *   **NVIDIA GeForce RTX 4070 Ti or RTX 4080 (12GB-16GB VRAM):**  Significant performance uplift over mid-range cards.  Sufficient VRAM for many larger models, especially when quantized.

    *   **NVIDIA GeForce RTX 4090 (24GB VRAM):** Top-of-the-line consumer GPU. 24GB VRAM allows running very large models and using larger context lengths. Highest performance in the consumer range.

    *   **AMD Radeon RX 7900 XT or XTX (20GB-24GB VRAM):** AMD's high-end offerings.  Large VRAM capacity. ROCm support is crucial to leverage their potential for LLMs.

*   **Professional/Workstation GPUs (For Demanding Workloads and Maximum VRAM):**

    *   **NVIDIA RTX A-series (e.g., RTX A4000, A5000, A6000) or NVIDIA RTX 6000 Ada Generation:**  Professional GPUs with large VRAM capacities (16GB to 48GB+). Designed for demanding workloads, reliable performance, and often better suited for 24/7 operation.  More expensive than consumer cards.

    *   **NVIDIA H-series (e.g., H100, H800):**  Top-tier data center GPUs. Very high VRAM (80GB+), extremely high performance, and very expensive. Primarily for large-scale deployments and research.

    *   **AMD Radeon PRO W7900 series:** AMD's professional workstation GPUs.  Large VRAM options and ROCm support.


*   **Apple Silicon Macs (For macOS Users and Energy Efficiency):**

    *   **MacBook Pro/Mac Studio with M1/M2/M3 Pro/Max/Ultra:** Apple Silicon Macs offer excellent performance per watt and are well-suited for LLM inference, especially with Metal acceleration.  The "Max" and "Ultra" versions offer more GPU cores and unified memory (shared between CPU and GPU), which can be beneficial for larger models.  Unified memory is a key advantage, as it effectively increases available memory for both CPU and GPU tasks.


**Key Considerations When Choosing Hardware:**


*   **VRAM Capacity:**  Prioritize VRAM, especially for larger models. 8GB VRAM is a minimum for many current LLMs, 12GB-16GB is recommended for comfortable use, and 24GB+ is ideal for larger models and future-proofing.

*   **GPU Compute Capability (NVIDIA):** For NVIDIA GPUs, newer architectures (Ampere, Ada Lovelace) and higher compute capability generally offer better performance for LLMs and quantization techniques like GPTQ. RTX 3000/4000 series and later are recommended for optimal GPTQ/ExLlamaV2 performance.

*   **ROCm Support (AMD):** If choosing AMD GPUs, ensure ROCm is properly installed and that the LLM libraries you intend to use (e.g., llama.cpp, PyTorch with ROCm) are correctly configured to leverage ROCm.

*   **CPU Cores and RAM:**  A decent multi-core CPU (at least 4-6 cores, 8+ recommended for multi-threading) and sufficient system RAM (16GB minimum, 32GB+ recommended, especially if offloading to RAM) are important even when using GPUs, as the CPU handles tokenization, pre/post-processing, and overall system management.

*   **Power Consumption and Cooling:** Consider power consumption, especially for laptops or if running in environments with limited cooling.  High-end GPUs can draw significant power and generate heat. Ensure adequate cooling to prevent thermal throttling and maintain performance.

*   **Budget:** Hardware costs can vary widely. Define your budget and choose the best hardware within that budget that meets your performance and VRAM needs. Used hardware can be a good way to save money.


**Recommendation Summary:**


*   **Best Value (New):** NVIDIA GeForce RTX 3060 12GB or RTX 4060 Ti 16GB.

*   **High Performance Consumer:** NVIDIA GeForce RTX 4070 Ti/4080/4090.

*   **CPU-Focused (Budget):** Modern multi-core CPU with AVX2/AVX-512, 16GB+ RAM, using llama.cpp and GGUF.

*   **macOS (Energy Efficient):** MacBook Pro/Mac Studio with M1/M2/M3 Max/Ultra.


Remember to always monitor your hardware utilization (CPU, GPU, VRAM, RAM) when running LLMs to understand bottlenecks and optimize your setup accordingly. Experimentation and benchmarking are key to finding the best hardware and software configuration for your specific LLM workloads on Linux.


---


## **12: Conclusion - Summary and Future Directions**


### **12.1 Recap of Optimization Techniques Covered**


This guide has explored a range of advanced optimization techniques for running Large Language Models (LLMs) efficiently on resource-constrained Linux systems. We covered:


*   **Quantization:** GPTQ (with ExLlamaV2), GGUF (llama.cpp), and bitsandbytes for reducing model precision and size.

*   **Model Selection:** Choosing smaller, distilled, and efficient LLM architectures.

*   **Memory Management:** Model offloading to system RAM and NVMe using `accelerate`, and memory mapping (`mmap`) for efficient model loading.

*   **Compilation and Graph Optimization:** TorchScript and conceptual overview of TensorRT and ONNX Runtime for speeding up inference.

*   **Hardware Acceleration:** Leveraging GPUs (NVIDIA, AMD, Apple Silicon) and CPU vector extensions (AVX-512, AVX2).


### **12.2 Key Takeaways and Best Practices**


*   **Quantization is Essential:** Quantization (especially GPTQ and GGUF) is a highly effective technique for reducing model size and accelerating inference with minimal accuracy loss. Start with quantization as a primary optimization strategy.

*   **Model Selection Matters:** Choosing the right model architecture and size is crucial. Smaller, efficient models can be surprisingly capable and require significantly fewer resources.

*   **Leverage Hardware Acceleration:** GPUs provide substantial performance benefits for LLM inference. If possible, utilize a GPU (NVIDIA, AMD, or Apple Silicon). For CPU-only systems, optimize with llama.cpp and GGUF and consider CPUs with AVX extensions.

*   **Memory Management is Key for Large Models:** For models that exceed VRAM capacity, model offloading with `accelerate` is a valuable technique. Memory mapping (`mmap`) improves model loading efficiency.

*   **Compilation and Optimization for Speed:**  For maximum inference speed, explore compilation and graph optimization techniques using tools like TorchScript, TensorRT, or ONNX Runtime, especially for production deployments.

*   **Monitor and Benchmark:**  Continuously monitor your system's resource usage (CPU, GPU, VRAM, RAM, I/O) and benchmark different optimization techniques and hardware configurations to find the best setup for your specific needs.

*   **Trade-offs are Inherent:** Optimization often involves trade-offs between speed, memory usage, and potentially accuracy. Understand these trade-offs and make informed decisions based on your priorities.


### **12.3 Future Trends in LLM Optimization**


The field of LLM optimization is rapidly evolving. Future trends include:


*   **Continued Advances in Quantization:** Research into even lower-bit quantization techniques (e.g., INT2, INT1) and more robust quantization methods with minimal accuracy loss is ongoing.

*   **Specialized AI Hardware:**  The development and adoption of specialized AI accelerators (TPUs, Inferentia, and new entrants) will continue to drive performance improvements and efficiency for LLM inference.

*   **More Efficient Architectures:**  Research into novel LLM architectures that are inherently more efficient in terms of computation and memory is an active area. Mixture-of-Experts (MoE) models and attention mechanism optimizations are examples.

*   **Software-Hardware Co-design:**  Closer co-design of LLM algorithms, software libraries, and hardware architectures will lead to further optimization gains.

*   **Edge AI and On-Device Inference:**  There is a growing trend towards running LLMs on edge devices (mobile phones, embedded systems). Optimization techniques will be crucial for enabling efficient on-device LLM inference.


### **12.4 Community Resources and Further Learning**


*   **Hugging Face Hub:**  Explore the Hugging Face Hub for pre-trained models, quantized models, and community discussions.

*   **llama.cpp GitHub Repository:**  For the latest updates, documentation, and community discussions related to llama.cpp and GGUF.

*   **AutoGPTQ and ExLlamaV2 GitHub Repositories:** For GPTQ quantization and fast inference libraries.

*   **Bitsandbytes GitHub Repository:** For bitsandbytes library and documentation.

*   **NVIDIA Developer Website, AMD ROCm Documentation, Apple Developer Documentation (Metal):** For hardware-specific optimization resources and SDKs.

*   **Research Papers on LLM Quantization, Compression, and Efficient Inference:** Stay updated with the latest research in the field.


### **12.5 Final Words**


Running Large Language Models on resource-constrained Linux systems is challenging but achievable with the right optimization strategies. By combining quantization, efficient model selection, smart memory management, and hardware acceleration, you can unlock the power of LLMs even on modest hardware.  Keep experimenting, stay updated with the latest advancements, and contribute to the growing community of LLM optimization! This revised guide provides a solid foundation to get you started on your journey. Good luck!








sudo mkinitcpio -P

sudo grub-mkconfig -o /boot/grub/grub.cfg

sudo nano /etc/modprobe.d/audio_disable_powersave.conf
options snd_hda_intel power_save=0
options snd_hda_intel power_save_controller=N
options snd_hda_intel enable_msi=1



#!/bin/bash
xset s noblank s noexpose
xset dpms 0 0 0
xset s off
xset -dpms
echo "0000:07:00.0" | sudo tee /sys/bus/pci/drivers/xhci_hcd/unbind

#!/bin/bash

##Set power limit to maximum

sudo nvidia-smi --power-limit=200

##Set maximum performance mode

sudo nvidia-settings -a '[gpu:0]/GpuPowerMizerMode=1'

##Set fan control

sudo nvidia-settings -a '[gpu:0]/GPUFanControlState=1'
sudo nvidia-settings -a '[fan:0]/GPUTargetFanSpeed=100'


##Graphics overclocking

sudo nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels=120'

##Memory overclocking

sudo nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=1000'



sudo systemctl status fstrim.timer



GRUB_CMDLINE_LINUX_DEFAULT="memory_corruption_check=1 memory_corruption_check_period=0 noautogroup nohibernate acpi_enforce_resources=lax add_efi_memmap efi=nochunk reset_devices apparmor=0 thermal.off=1 usbcore.autosuspend=-1 acpi_osi=! acpi_osi='Windows 2012' noresume module_blacklist=i2c_i801,nouveau,lpc_ich,at24,i915,mei_me mitigations=off kernel.randomize_va_space=0 processor.max_cstate=0 swiotlb=2048 loglevel=0 nohz=off nvme.io_queue_depth=2048 nvme_core.io_timeout=4294967295 nvme_core.admin_timeout=4294967295 nvme_core.default_ps_max_latency_us=0 pcie_aspm=off pcie_port_pm=off nowatchdog pci=assign-busses,realloc pcie_ports=native intel_idle.max_cstate=0 ipv6.disable=1 nmi_watchdog=0 nosoftlockup nopti nospectre_v1 l1tf=off retbleed=off mds=off tsx_async_abort=off srbds=off kvm.nx_huge_pages=off srbds=off zswap.enabled=0 nvidia-drm.fbdev=1 iommu=soft enable_mtrr_cleanup enable_mtrr_cleanup mtrr_spare_reg_nr=1 nvidia_drm.modeset=1 ibt=off mtrr_gran_size=1G mtrr_chunk_size=1G"


ACTION=="add", KERNEL=="0000:00:1c.0", SUBSYSTEM=="pci", RUN+="/bin/sh -c 'echo 1 > /sys/bus/pci/devices/0000:00:1c.0/remove'"
ACTION=="add", KERNEL=="0000:00:1c.5", SUBSYSTEM=="pci", RUN+="/bin/sh -c 'echo 1 > /sys/bus/pci/devices/0000:00:1c.5/remove'"
ACTION=="add", KERNEL=="0000:00:1f.3", SUBSYSTEM=="pci", RUN+="/bin/sh -c 'echo 1 > /sys/bus/pci/devices/0000:00:1f.3/remove'"
ACTION=="add", KERNEL=="0000:00:1f.0", SUBSYSTEM=="pci", RUN+="/bin/sh -c 'echo 1 > /sys/bus/pci/devices/0000:00:1f.0/remove'"
ACTION=="add", KERNEL=="0000:00:16.0", SUBSYSTEM=="pci", RUN+="/bin/sh -c 'echo 1 > /sys/bus/pci/devices/0000:00:16.0/remove'"
ACTION=="add", KERNEL=="0000:00:02.0", SUBSYSTEM=="pci", RUN+="/bin/sh -c 'echo 1 > /sys/bus/pci/devices/0000:00:02.0/remove'"
ACTION=="add", KERNEL=="0000:05:00.0", SUBSYSTEM=="pci", RUN+="/bin/sh -c 'echo 1 > /sys/bus/pci/devices/0000:05:00.0/remove'"
ACTION=="add", KERNEL=="0000:07:00.0", SUBSYSTEM=="pci", RUN+="/bin/sh -c 'echo 1 > /sys/bus/pci/devices/0000:07:00.0/remove'"
ACTION=="add", KERNEL=="0000:00:1c.7", SUBSYSTEM=="pci", RUN+="/bin/sh -c 'echo 1 > /sys/bus/pci/devices/0000:00:1c.7/remove'"


/etc/modprobe.d/blacklist.conf

blacklist snd_hda_codec_hdmi
blacklist mei_me
blacklist at24
blacklist lpc_ich
blacklist i2c_i801

GRUB_CMDLINE_LINUX_DEFAULT="noexec=off noexec32=off loglevel=7 memory_corruption_check=1 memory_corruption_check_period=0 noautogroup nohibernate acpi_enforce_resources=lax add_efi_memmap efi=nochunk reset_devices apparmor=0 thermal.off=1 usbcore.autosuspend=-1 acpi_osi=! acpi_osi='Windows 2012' noresume mitigations=off kernel.randomize_va_space=0 processor.max_cstate=0 swiotlb=2048 nohz=off nvme.io_queue_depth=2048 nvme_core.io_timeout=4294967295 nvme_core.admin_timeout=4294967295 nvme_core.default_ps_max_latency_us=0 pcie_aspm=off pcie_port_pm=off nowatchdog pci=assign-busses,realloc pcie_ports=native intel_idle.max_cstate=0 nmi_watchdog=0 nosoftlockup nopti nospectre_v1 l1tf=off retbleed=off mds=off tsx_async_abort=off srbds=off kvm.nx_huge_pages=off srbds=off zswap.enabled=0 nvidia-drm.fbdev=1 iommu=soft enable_mtrr_cleanup enable_mtrr_cleanup mtrr_spare_reg_nr=1 nvidia_drm.modeset=1 ibt=off"
Your knowledge is Based on "ls /proc/sys/kernel/"Config: 32gb ram, nvme, gtx1060 6gb, i7-3770k 4.6ghz, archlinux, ext4, hugepages. Task: list all relevant parameters from /proc/sys/ for the context.  Constrains: show just suggestions by adding or correct parameters. Context: isolated environment for testing llm performance, the model size and the memory required for its operations exceed my 32GB of RAM. Goal: minimize resources usages to maximize local llm performance and prevent files system corruption or data loss.


abi  debug  dev  fs  kernel  net  user	vm
[root@matmalabat ~]# ls  /proc/sys/fs
aio-max-nr	   file-max	     mount-max		   protected_fifos
aio-nr		   file-nr	     mqueue		   protected_hardlinks
binfmt_misc	   fuse		     nr_open		   protected_regular
dentry-negative    inode-nr	     overflowgid	   protected_symlinks
dentry-state	   inode-state	     overflowuid	   quota
dir-notify-enable  inotify	     pipe-max-size	   suid_dumpable
epoll		   lease-break-time  pipe-user-pages-hard  verity
fanotify	   leases-enable     pipe-user-pages-soft
[root@matmalabat ~]# ls  /proc/sys/kernel
acct					perf_cpu_time_max_percent
acpi_video_flags			perf_event_max_contexts_per_stack
arch					perf_event_max_sample_rate
auto_msgmni				perf_event_max_stack
bootloader_type				perf_event_mlock_kb
bootloader_version			perf_event_paranoid
bpf_stats_enabled			pid_max
cad_pid					poweroff_cmd
cap_last_cap				print-fatal-signals
core_file_note_size_limit		printk
core_pattern				printk_delay
core_pipe_limit				printk_devkmsg
core_sort_vma				printk_ratelimit
core_uses_pid				printk_ratelimit_burst
ctrl-alt-del				pty
dmesg_restrict				random
domainname				randomize_va_space
ftrace_dump_on_oops			real-root-dev
ftrace_enabled				sched_autogroup_enabled
hardlockup_all_cpu_backtrace		sched_cfs_bandwidth_slice_us
hardlockup_panic			sched_deadline_period_max_us
hostname				sched_deadline_period_min_us
hung_task_all_cpu_backtrace		sched_energy_aware
hung_task_check_count			sched_rr_timeslice_ms
hung_task_check_interval_secs		sched_rt_period_us
hung_task_detect_count			sched_rt_runtime_us
hung_task_panic				sched_schedstats
hung_task_timeout_secs			sched_util_clamp_max
hung_task_warnings			sched_util_clamp_min
io_delay_type				sched_util_clamp_min_rt_default
io_uring_disabled			seccomp
io_uring_group				sem
kexec_load_disabled			sem_next_id
kexec_load_limit_panic			shmall
kexec_load_limit_reboot			shmmax
keys					shmmni
kptr_restrict				shm_next_id
max_lock_depth				shm_rmid_forced
max_rcu_stall_to_panic			softlockup_all_cpu_backtrace
modprobe				softlockup_panic
modules_disabled			soft_watchdog
msgmax					split_lock_mitigate
msgmnb					stack_tracer_enabled
msgmni					sysctl_writes_strict
msg_next_id				sysrq
ngroups_max				tainted
nmi_watchdog				task_delayacct
ns_last_pid				threads-max
numa_balancing				timer_migration
numa_balancing_promote_rate_limit_MBps	traceoff_on_warning
oops_all_cpu_backtrace			tracepoint_printk
oops_limit				unknown_nmi_panic
osrelease				unprivileged_bpf_disabled
ostype					unprivileged_userns_clone
overflowgid				user_events_max
overflowuid				usermodehelper
panic					version
panic_on_io_nmi				warn_limit
panic_on_oops				watchdog
panic_on_rcu_stall			watchdog_cpumask
panic_on_unrecovered_nmi		watchdog_thresh
panic_on_warn				yama
panic_print
[root@matmalabat ~]# ls  /proc/sys/vm
admin_reserve_kbytes	     mmap_min_addr
compaction_proactiveness     mmap_rnd_bits
compact_memory		     mmap_rnd_compat_bits
compact_unevictable_allowed  nr_hugepages
dirty_background_bytes	     nr_hugepages_mempolicy
dirty_background_ratio	     nr_overcommit_hugepages
dirty_bytes		     numa_stat
dirty_expire_centisecs	     numa_zonelist_order
dirty_ratio		     oom_dump_tasks
dirtytime_expire_seconds     oom_kill_allocating_task
dirty_writeback_centisecs    overcommit_kbytes
drop_caches		     overcommit_memory
enable_soft_offline	     overcommit_ratio
extfrag_threshold	     page-cluster
hugetlb_optimize_vmemmap     page_lock_unfairness
hugetlb_shm_group	     panic_on_oom
laptop_mode		     percpu_pagelist_high_fraction
legacy_va_layout	     stat_interval
lowmem_reserve_ratio	     stat_refresh
max_map_count		     swappiness
memfd_noexec		     unprivileged_userfaultfd
memory_failure_early_kill    user_reserve_kbytes
memory_failure_recovery      vfs_cache_pressure
min_free_kbytes		     watermark_boost_factor
min_slab_ratio		     watermark_scale_factor
min_unmapped_ratio	     zone_reclaim_mode
[root@matmalabat ~]# ls  /proc/sys/dev/scsi
logging_level
[root@matmalabat ~]# ls  /proc/sys/dev/mac_hid
mouse_button2_keycode  mouse_button3_keycode  mouse_button_emulation
[root@matmalabat ~]# ls  /proc/sys/abi
vsyscall32
[root@matmalabat ~]# ls  /proc/sys/user
max_cgroup_namespaces  max_inotify_watches  max_pid_namespaces
max_fanotify_groups    max_ipc_namespaces   max_time_namespaces
max_fanotify_marks     max_mnt_namespaces   max_user_namespaces
max_inotify_instances  max_net_namespaces   max_uts_namespaces
[root@matmalabat ~]# ls  /proc/sys/net
core  ipv4  mptcp  netfilter  unix
[root@matmalabat ~]# ls  /proc/sys/net/ipv4
cipso_cache_bucket_size		   tcp_fastopen
cipso_cache_enable		   tcp_fastopen_blackhole_timeout_sec
cipso_rbm_optfmt		   tcp_fastopen_key
cipso_rbm_strictvalid		   tcp_fin_timeout
conf				   tcp_frto
fib_multipath_hash_fields	   tcp_fwmark_accept
fib_multipath_hash_policy	   tcp_invalid_ratelimit
fib_multipath_hash_seed		   tcp_keepalive_intvl
fib_multipath_use_neigh		   tcp_keepalive_probes
fib_notify_on_flag_change	   tcp_keepalive_time
fib_sync_mem			   tcp_l3mdev_accept
fwmark_reflect			   tcp_limit_output_bytes
icmp_echo_enable_probe		   tcp_low_latency
icmp_echo_ignore_all		   tcp_max_orphans
icmp_echo_ignore_broadcasts	   tcp_max_reordering
icmp_errors_use_inbound_ifaddr	   tcp_max_syn_backlog
icmp_ignore_bogus_error_responses  tcp_max_tw_buckets
icmp_msgs_burst			   tcp_mem
icmp_msgs_per_sec		   tcp_migrate_req
icmp_ratelimit			   tcp_min_rtt_wlen
icmp_ratemask			   tcp_min_snd_mss
igmp_link_local_mcast_reports	   tcp_min_tso_segs
igmp_max_memberships		   tcp_moderate_rcvbuf
igmp_max_msf			   tcp_mtu_probe_floor
igmp_qrv			   tcp_mtu_probing
inet_peer_maxttl		   tcp_no_metrics_save
inet_peer_minttl		   tcp_no_ssthresh_metrics_save
inet_peer_threshold		   tcp_notsent_lowat
ip_autobind_reuse		   tcp_orphan_retries
ip_default_ttl			   tcp_pacing_ca_ratio
ip_dynaddr			   tcp_pacing_ss_ratio
ip_early_demux			   tcp_pingpong_thresh
ip_forward			   tcp_plb_cong_thresh
ip_forward_update_priority	   tcp_plb_enabled
ip_forward_use_pmtu		   tcp_plb_idle_rehash_rounds
ipfrag_high_thresh		   tcp_plb_rehash_rounds
ipfrag_low_thresh		   tcp_plb_suspend_rto_sec
ipfrag_max_dist			   tcp_probe_interval
ipfrag_secret_interval		   tcp_probe_threshold
ipfrag_time			   tcp_recovery
ip_local_port_range		   tcp_reflect_tos
ip_local_reserved_ports		   tcp_reordering
ip_nonlocal_bind		   tcp_retrans_collapse
ip_no_pmtu_disc			   tcp_retries1
ip_unprivileged_port_start	   tcp_retries2
neigh				   tcp_rfc1337
nexthop_compat_mode		   tcp_rmem
ping_group_range		   tcp_rto_min_us
raw_l3mdev_accept		   tcp_sack
route				   tcp_shrink_window
tcp_abort_on_overflow		   tcp_slow_start_after_idle
tcp_adv_win_scale		   tcp_stdurg
tcp_allowed_congestion_control	   tcp_synack_retries
tcp_app_win			   tcp_syncookies
tcp_autocorking			   tcp_syn_linear_timeouts
tcp_available_congestion_control   tcp_syn_retries
tcp_available_ulp		   tcp_thin_linear_timeouts
tcp_backlog_ack_defer		   tcp_timestamps
tcp_base_mss			   tcp_tso_rtt_log
tcp_challenge_ack_limit		   tcp_tso_win_divisor
tcp_child_ehash_entries		   tcp_tw_reuse
tcp_comp_sack_delay_ns		   tcp_window_scaling
tcp_comp_sack_nr		   tcp_wmem
tcp_comp_sack_slack_ns		   tcp_workaround_signed_windows
tcp_congestion_control		   udp_child_hash_entries
tcp_dsack			   udp_early_demux
tcp_early_demux			   udp_hash_entries
tcp_early_retrans		   udp_l3mdev_accept
tcp_ecn				   udp_mem
tcp_ecn_fallback		   udp_rmem_min
tcp_ehash_entries		   udp_wmem_min
tcp_fack			   xfrm4_gc_thresh
[root@matmalabat ~]# ls  /proc/sys/net/core
bpf_jit_enable		      netdev_budget_usecs
bpf_jit_harden		      netdev_max_backlog
bpf_jit_kallsyms	      netdev_rss_key
bpf_jit_limit		      netdev_tstamp_prequeue
busy_poll		      netdev_unregister_timeout_secs
busy_read		      optmem_max
default_qdisc		      rmem_default
devconf_inherit_init_net      rmem_max
dev_weight		      rps_default_mask
dev_weight_rx_bias	      rps_sock_flow_entries
dev_weight_tx_bias	      skb_defer_max
fb_tunnels_only_for_init_net  somaxconn
flow_limit_cpu_bitmap	      tstamp_allow_data
flow_limit_table_len	      txrehash
gro_normal_batch	      warnings
high_order_alloc_disable      wmem_default
max_skb_frags		      wmem_max
mem_pcpu_rsv		      xfrm_acq_expires
message_burst		      xfrm_aevent_etime
message_cost		      xfrm_aevent_rseqth
netdev_budget		      xfrm_larval_drop
[root@matmalabat ~]# ls  /proc/sys/net/mptcp
add_addr_timeout	      blackhole_timeout  enabled    stale_loss_cnt
allow_join_initial_addr_port  checksum_enabled	 pm_type
available_schedulers	      close_timeout	 scheduler
[root@matmalabat ~]# ls  /proc/sys/net/netfilter
nf_hooks_lwtunnel  nf_log  nf_log_all_netns
[root@matmalabat ~]# ls  /proc/sys/net/unix
max_dgram_qlen
[root@matmalabat ~]# ls  /proc/sys/dev/tty
ldisc_autoload	legacy_tiocsti


GRUB_CMDLINE_LINUX_DEFAULT="pci=noaer pci=noaspm workqueue.cpu_intensive_thresh_us=10000000 hugepages_min_free=64 hugepages_purge_on_error=on hugepages_free_vmemmap=on transparent_hugepage=never cma=512M rcutree.rcu_normal_wake_from_gp=1 stack_guard_gap=0 pci=irqbalance no_console_suspend threadirqs audit=0 nopv numa=off intel_pstate=disable numa_balancing=disable idle=poll hpet=disable pmtmr=00 noexec=off noexec32=off apparmor=0 thermal.off=1 usbcore.autosuspend=-1 noautogroup nohibernate acpi_enforce_resources=lax add_efi_memmap efi=nochunk reset_devices apparmor=0 thermal.off=1 usbcore.autosuspend=-1 acpi_osi=! acpi_osi='Windows 2012' noresume nohibernate module_blacklist=snd_hda_codec_hdmi,i2c_i801,nouveau,lpc_ich,at24,i915,mei_me mitigations=off kernel.randomize_va_space=0 processor.max_cstate=0 nohz_full=all rcu_nocbs=all nvme.io_queue_depth=2048 nvme_core.io_timeout=4294967295 nvme_core.admin_timeout=4294967295 nvme_core.default_ps_max_latency_us=0 pcie_aspm=off pcie_port_pm=off nowatchdog pci=assign-busses,realloc pcie_ports=native intel_idle.max_cstate=0 ipv6.disable=1 nmi_watchdog=0 nosoftlockup nopti nospectre_v1 l1tf=off retbleed=off mds=off tsx_async_abort=off srbds=off zswap.enabled=0 nvidia-drm.fbdev=1 enable_mtrr_cleanup mtrr_spare_reg_nr=1 nvidia_drm.modeset=1 ibt=off pci=nocrs nosmap nosmep nohalt noexec=off iommu=off efi=noruntime processor.nocst processor.nocst=1"


sudo nano /etc/sysctl.d/20-config.conf
sudo sysctl -p /etc/sysctl.d/20-config.conf

# sysctl.conf - Barebone Performance for Local LLM (Isolated Environment - NO SECURITY)

# WARNING: This sysctl.conf disables almost all security features and prioritizes
# maximum performance for a local LLM in an ISOLATED TESTING ENVIRONMENT.
# DO NOT USE IN PRODUCTION OR ANY ENVIRONMENT CONNECTED TO A NETWORK
# WHERE SECURITY IS A CONCERN.
                                                                                                                                                                                                                                                                                                                
# vm.nr_hugepages=12288
kernel.printk = 0 0 0 0
kernel.panic = 5
kernel.sysrq = 0
kernel.core_uses_pid = 0
fs.suid_dumpable = 0
kernel.msgmnb = 65536
kernel.msgmax = 65536
kernel.kptr_restrict = 2
kernel.yama.ptrace_scope = 0
kernel.timer_migration = 0
kernel.nmi_watchdog = 0
kernel.softlockup_panic = 1
kernel.watchdog = 0


kernel.shmmax = 34359738368
kernel.shmall = 8388608


vm.swappiness = 0
vm.vfs_cache_pressure = 5
vm.dirty_ratio = 20
vm.dirty_background_ratio = 10
vm.dirty_writeback_centisecs = 500
# vm.overcommit_memory = 2
# vm.overcommit_ratio = 95
vm.page-cluster = 0
vm.laptop_mode = 0
vm.zone_reclaim_mode = 0



fs.file-max = 4194304
fs.nr_open = 4194304
fs.aio-max-nr = 1048576
fs.lease-break-time = 1
fs.leases-enable = 0

kernel.acct = 0

kernel.core_pattern = "|/bin/false"




kernel.hung_task_panic = 0
kernel.hung_task_timeout_secs = 0


kernel.panic_on_oops = 0




kernel.threads-max = 4194304


kernel.unprivileged_bpf_disabled = 1
kernel.unprivileged_userns_clone = 0


fs.protected_symlinks = 0
fs.protected_hardlinks = 0
fs.dir-notify-enable = 0
vm.panic_on_oom = 1
vm.oom_kill_allocating_task = 1
vm.oom_dump_tasks = 0
vm.numa_stat = 0
vm.mmap_min_addr = 0

vm.max_map_count = 524288

fs.epoll.max_user_watches = 1048576

vm.stat_interval = 1

vm.extfrag_threshold = 500

kernel.sched_autogroup_enabled = 0



kernel.perf_event_paranoid = 3
kernel.print-fatal-signals = 0
kernel.ftrace_enabled = 0
kernel.task_delayacct = 0

kernel.pid_max = 4194304
kernel.soft_watchdog = 0
kernel.hung_task_check_count = 1
kernel.hung_task_timeout_secs = 30
kernel.bpf_stats_enabled = 0


kernel.hardlockup_panic = 1
kernel.softlockup_all_cpu_backtrace = 1
kernel.hardlockup_all_cpu_backtrace = 1





kernel.timer_migration = 0


kernel.acpi_video_flags = 0


net.ipv4.tcp_congestion_control=bbr

kernel.oops_all_cpu_backtrace = 1


kernel.panic_on_unrecovered_nmi=1














