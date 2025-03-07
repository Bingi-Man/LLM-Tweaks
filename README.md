## Procedures Guide: Advanced LLM Inference Optimization on Linux (Sections 4-8)

This guide provides step-by-step procedures for implementing advanced optimization techniques for running Large Language Models (LLMs) efficiently on resource-constrained Linux systems. It covers techniques from sections 4 through 8 of the main document, focusing on practical implementation and including all relevant code snippets.

---

#### Procedure 4: Running GPTQ Quantized Models with ExLlamaV2 for High-Performance Inference

**Goal:**  Achieve fast and memory-efficient LLM inference on NVIDIA GPUs using GPTQ quantization and the ExLlamaV2 library.

**Prerequisites:**

*   **NVIDIA GPU:** RTX 3000/4000 series or later recommended for optimal ExLlamaV2 performance.
*   **CUDA Drivers:**  Properly installed and configured NVIDIA CUDA drivers.
*   **Python Environment:** Python 3.8 or higher with `pip` package manager.

**Steps:**

1.  **Install Required Python Libraries:**

    Open a terminal and execute the following command to install `auto-gptq`, `optimum`, and `transformers`:

    ```bash
    pip install auto-gptq optimum transformers
    ```

    These libraries provide the necessary tools for loading and running GPTQ quantized models from Hugging Face Hub.

2.  **Select a GPTQ Quantized Model from Hugging Face Hub:**

    Browse the Hugging Face Hub ([https://huggingface.co/models](https://huggingface.co/models)) and search for models quantized using GPTQ. Look for models tagged with "GPTQ" or published by users known for quantized models (e.g., "TheBloke").

    **Important Information to Note from the Model Card:**

    *   **`model_name_or_path`:** The full model identifier (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"). You will use this to load the model.
    *   **`model_basename`:**  The base filename of the GPTQ quantized model files. This is often mentioned in the model card or repository files. For example, for "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", the `model_basename` is typically "gptq_model-4bit-128g".

3.  **Prepare Python Inference Script:**

    Create a Python file (e.g., `gptq_inference.py`) and copy the following code into it.  **Replace the placeholder values** for `model_name_or_path` and `model_basename` with the information you noted from the Hugging Face Hub model card in the previous step.

    ```python
    # Install AutoGPTQ and optimum (if not already installed)
    # pip install auto-gptq
    # pip install optimum  # For safetensors support

    from auto_gptq import AutoGPTQForCausalLM
    from transformers import AutoTokenizer
    import torch
    import os

    model_name_or_path = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"  # <-- REPLACE WITH YOUR MODEL PATH
    model_basename = "gptq_model-4bit-128g"  # <-- REPLACE WITH YOUR MODEL BASE NAME

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # Load the quantized model
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,  # Recommended for faster and safer loading of model weights
        trust_remote_code=True,  # Required for some models
        device="cuda:0",  # Use the first GPU if available, or "cpu" to force CPU usage
        use_triton=False,  # Set to True if you have Triton installed for potentially faster inference (requires Triton installation)
        quantize_config=None,  # Set to None when loading a pre-quantized GPTQ model
    )

    # Example prompt
    prompt = "Write a short story about a cat who learns to code."

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")  # Move input tensors to GPU if model is on GPU

    # Generate text
    with torch.no_grad():  # Disable gradient calculation for inference
        generation_output = model.generate(
            **inputs,  # **inputs unpacks the dictionary returned by the tokenizer
            max_new_tokens=100,  # Maximum number of tokens to generate
            do_sample=True,      # Enable sampling for more diverse outputs
            top_k=50,           # Top-k sampling parameter
            top_p=0.95,          # Top-p (nucleus) sampling parameter
            temperature=0.7,     # Sampling temperature (higher = more random)
        )

    # Decode the output
    generated_text = tokenizer.decode(generation_output[0])
    print(generated_text)

    # Print model size
    print(f"Number of parameters: {model.num_parameters()}")
    model_size_mb = os.path.getsize(os.path.join(model_name_or_path, model_basename + ".safetensors")) / (1024**2)  # Assuming safetensors
    print(f"Quantized model size: {model_size_mb:.2f} MB")  # Or GB if larger
    ```

4.  **Run the Inference Script:**

    Execute the Python script from your terminal:

    ```bash
    python gptq_inference.py
    ```

    This will load the GPTQ quantized model onto your GPU (or CPU if `device="cpu"` is specified), tokenize the example prompt, generate text, and print the generated output along with model size information.

5.  **Monitor VRAM Usage (Optional but Recommended):**

    While the script is running, open another terminal and use the `nvidia-smi` command to monitor your GPU VRAM usage:

    ```bash
    nvidia-smi
    ```

    Observe the VRAM utilization to ensure you are not exceeding your GPU's memory capacity. GPTQ quantization should significantly reduce VRAM usage compared to the full-precision model.

**Troubleshooting GPTQ Loading Errors:**

*   **"CUDA out of memory" Error:**
    *   **Solution:**  Reduce `max_new_tokens` in the `model.generate()` call to generate shorter sequences. Try using a smaller GPTQ quantized model. If using `llama.cpp` (GPTQ support is less common there), consider offloading more layers to the CPU using the `-ngl` parameter. Close any other applications that are using GPU VRAM.

*   **"ModuleNotFoundError: No module named 'auto_gptq'" or "'optimum'" Error:**
    *   **Solution:** Ensure you have installed the necessary libraries correctly. Double-check your Python environment and re-run the `pip install auto-gptq optimum transformers` command.

*   **Model Loading Errors (e.g., "ValueError: ...") Error:**
    *   **Solution:** Carefully verify the `model_name_or_path` and `model_basename` values in your Python script against the information provided on the Hugging Face Hub model card for the GPTQ model you are trying to load. Ensure `use_safetensors=True` is set if the model uses the safetensors format (which is common and recommended). For some models, you might need to add `trust_remote_code=True` in the `from_quantized` function call; check the model card for specific instructions or warnings about remote code execution.

---

#### Procedure 5: Running GGUF Models with llama.cpp for CPU and Cross-Platform Efficiency

**Goal:** Run LLMs efficiently on CPUs and across different operating systems (Linux, macOS, Windows) using GGUF quantized models and the `llama.cpp` library.

**Prerequisites:**

*   **C++ Compiler and Build Tools:**  A C++ compiler (like g++) and `make` (or CMake) are required to build `llama.cpp`.
*   **CMake (Recommended):** CMake is recommended for a more robust and flexible build process.
*   **Optional: CUDA or ROCm (for GPU acceleration):** NVIDIA CUDA drivers or AMD ROCm drivers if you want to enable GPU acceleration with `llama.cpp`.

**Steps:**

1.  **Download and Build `llama.cpp`:**

    Open a terminal and execute the following commands to download and build the `llama.cpp` library from source:

    ```bash
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    ```

    **Choose Build Options based on your Hardware:**

    *   **CPU-Only Build (Default):** For basic CPU inference, simply run `make`:

        ```bash
        make
        ```

        This will build `llama.cpp` with default CPU optimizations using OpenBLAS for linear algebra.

    *   **NVIDIA GPU (CUDA) Build:** To enable NVIDIA GPU acceleration using CUDA, use the following `make` command. **Adjust `CUDA_DIR` to the correct path of your CUDA installation.**  You can check your CUDA installation path by running `nvcc --version`. Common paths are `/usr/local/cuda` or `/usr/cuda`.

        ```bash
        make LLAMA_CUDA=1 CUDA_DIR=/usr/local/cuda
        ```

    *   **AMD GPU (ROCm) Build:** To enable AMD GPU acceleration using ROCm, use the following `make` command. **Adjust `HIP_DIR` to the correct path of your ROCm installation.** You can check your ROCm installation path by running `hipcc --version`. A common path is `/opt/rocm`.

        ```bash
        make LLAMA_HIP=1 HIP_DIR=/opt/rocm
        ```

    *   **macOS Metal (Apple Silicon GPU) Build:** For macOS systems with Apple Silicon chips, enable Metal GPU acceleration with:

        ```bash
        make LLAMA_METAL=1
        ```

    After running the `make` command with your chosen options, the `llama.cpp` binaries (including `./main`) will be built in the main `llama.cpp` directory.

2.  **Download a GGUF Quantized Model:**

    Go to the Hugging Face Hub and search for models in the GGUF format.  "TheBloke" is a popular user who provides many models in GGUF format.  Download a GGUF model file (e.g., a `.gguf` file, often with quantization level in the filename like `Q4_K_M`).

    Create a `models` directory inside the `llama.cpp` directory:

    ```bash
    mkdir models
    ```

    Then, use `wget` (or your preferred download method) to download the GGUF model file into the `models` directory. For example, to download "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf":

    ```bash
    wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    ```

3.  **Run Inference using `./main` Binary:**

    Navigate back to the main `llama.cpp` directory in your terminal.  Execute the `./main` binary to run inference. Here's an example command, **adjust the `-m` path to point to your downloaded GGUF model file**:

    ```bash
    ./main -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -n 100 -p "Write a short story about a robot..." -t 8 -ngl 20
    ```

    **Explanation of `llama.cpp` command-line parameters:**

    *   `./main`:  Executes the `main` binary.
    *   `-m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`:  Specifies the path to the GGUF model file using the `-m` (model) parameter. **Replace this path with the actual path to your downloaded GGUF model file.**
    *   `-n 100`:  Sets the maximum number of new tokens to generate to 100 using the `-n` (n_predict) parameter.
    *   `-p "Write a short story about a robot..."`:  Provides the initial prompt text using the `-p` (prompt) parameter.  **Change the prompt to your desired input text.**
    *   `-t 8`:  Sets the number of threads to use for CPU inference to 8 using the `-t` (threads) parameter. Adjust this based on your CPU core count. Start with the number of physical cores your CPU has.
    *   `-ngl 20`:  `-ngl` (number of GPU layers) is an *optional* parameter for GPU acceleration. It specifies how many layers of the neural network to offload to the GPU.  If you built `llama.cpp` with GPU support (CUDA, ROCm, or Metal), you can use `-ngl` to offload computation to the GPU, potentially speeding up inference.  The value `20` is an example; experiment with different values.  A higher value offloads more layers to the GPU. If you are running on CPU only, you can omit the `-ngl` parameter.

    **Adjust Parameters for Performance and Resource Usage:**

    *   Experiment with different values for `-n` (max tokens), `-t` (threads), and `-ngl` (GPU layers) to optimize inference speed and memory usage for your specific hardware and model.
    *   For CPU-only inference, `-t` (threads) is crucial for performance. Set it to the number of physical CPU cores or slightly higher.
    *   For GPU acceleration, `-ngl` controls how much computation is offloaded to the GPU. Increase `-ngl` to utilize the GPU more, but be mindful of VRAM usage.

**Troubleshooting `llama.cpp` Build Issues:**

*   **"make: command not found" or "g++: command not found" Error:**
    *   **Solution:** You are missing essential build tools. Install a C++ compiler (like g++) and `make`.  On Debian/Ubuntu-based systems, use: `sudo apt-get install build-essential cmake`. On Fedora/CentOS/RHEL-based systems, use: `sudo yum groupinstall "Development Tools" cmake`.

*   **"CMake Error: CMake was unable to find a build program..." Error:**
    *   **Solution:** CMake is not installed. Install CMake. On Debian/Ubuntu: `sudo apt-get install cmake`. On Fedora/CentOS/RHEL: `sudo yum install cmake`.

*   **CUDA/ROCm Build Errors (GPU acceleration issues):**
    *   **Solution:** Verify that CUDA or ROCm is correctly installed and configured on your system. Check the output of `nvcc --version` (for CUDA) or `hipcc --version` (for ROCm) to confirm installation and version. Ensure that the `CUDA_DIR` or `HIP_DIR` paths specified in your `make` command are accurate and point to your CUDA or ROCm installation directory. Refer to the `llama.cpp` README file on GitHub for detailed GPU build instructions and troubleshooting tips.

*   **"Illegal instruction" or Crashes at Runtime:**
    *   **Solution:** This error often indicates CPU architecture incompatibility or issues with the chosen GGUF quantization level. Try building `llama.cpp` with different CPU optimization flags (refer to the `llama.cpp` documentation for CPU-specific build flags). Alternatively, try using a different GGUF quantization level for the model (e.g., a less aggressively quantized version).

---

#### Procedure 6: Loading 4-bit Quantized Models with Bitsandbytes in Transformers for Easy Memory Reduction

**Goal:**  Easily load and run 4-bit quantized LLMs within the Hugging Face Transformers ecosystem using the `bitsandbytes` library to significantly reduce GPU memory usage.

**Prerequisites:**

*   **Python Environment:** Python 3.8 or higher with `pip` package manager.
*   **PyTorch:** PyTorch installed and configured for your hardware (CPU or GPU).
*   **Optional: CUDA (for GPU acceleration):** NVIDIA CUDA drivers if you want to use GPU acceleration.

**Steps:**

1.  **Install `bitsandbytes` and `transformers`:**

    Open a terminal and install the `bitsandbytes` and `transformers` libraries using pip:

    ```bash
    pip install bitsandbytes transformers
    ```

    Ensure that the installation completes without errors. If you encounter issues, check your Python environment and PyTorch installation.

2.  **Select a Model from Hugging Face Hub:**

    Choose a pre-trained model from the Hugging Face Hub that you want to load in 4-bit quantized format. You can start with a smaller model for initial testing, such as `facebook/opt-350m`.  Note the `model_id` (e.g., "facebook/opt-350m").

3.  **Create Python Inference Script:**

    Create a Python file (e.g., `bitsandbytes_inference.py`) and copy the following code into it. **Replace `"facebook/opt-350m"` with the `model_id` of your chosen model.**

    ```python
    # Install bitsandbytes and transformers (if not already installed)
    # pip install bitsandbytes
    # pip install transformers

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_id = "facebook/opt-350m"  # <-- REPLACE WITH YOUR MODEL ID from Hugging Face Hub

    # Load tokenizer (standard transformers way)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model in 4-bit using bitsandbytes
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        load_in_4bit=True,
        torch_dtype=torch.float16  # Load in 4-bit with float16 for mixed-precision operations
    )

    # Example prompt
    prompt = "Write a poem about the Linux operating system."

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")  # Move inputs to GPU

    # Generate text
    with torch.no_grad():
        outputs = model_4bit.generate(**inputs, max_new_tokens=50)

    # Decode and print output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    ```

4.  **Run the Inference Script:**

    Execute the Python script from your terminal:

    ```bash
    python bitsandbytes_inference.py
    ```

    This script will load the tokenizer and then load the specified model in 4-bit quantized format using `bitsandbytes`. The `device_map='auto'` argument will automatically place the model on the GPU if available, or fall back to the CPU if no GPU is detected. `load_in_4bit=True` activates 4-bit quantization, and `torch_dtype=torch.float16` is recommended for mixed-precision operations with 4-bit quantization, potentially improving accuracy.

5.  **Monitor Memory Usage:**

    To verify that 4-bit quantization is effectively reducing memory usage, monitor your GPU VRAM usage using `nvidia-smi` while the script is running. Compare the VRAM usage when loading the model with `load_in_4bit=True` to the VRAM usage when loading the same model without this argument (in full precision). You should observe a significant reduction in VRAM consumption when using 4-bit quantization.

---

#### Procedure 7: Model Selection for Efficient LLM Architectures

**Goal:** Choose the most efficient LLM architecture and model size for your specific resource constraints and performance needs.

**Considerations and Steps:**

1.  **Define Your Resource Constraints:**

    *   **VRAM:** How much GPU VRAM is available? (e.g., 4GB, 8GB, 12GB, 24GB).
    *   **System RAM:** How much system RAM is available? (e.g., 16GB, 32GB, 64GB).
    *   **CPU:** What type of CPU do you have? (Number of cores, clock speed, AVX support).
    *   **Inference Speed Requirements:** How quickly do you need the model to generate text? (Latency-sensitive application vs. throughput-focused).

2.  **Determine Desired Output Quality and Task Complexity:**

    *   **Task:** What type of task will the LLM be used for? (Chat, code generation, creative writing, information extraction, etc.).
    *   **Output Quality:** What level of coherence, fluency, and accuracy is required in the generated text? For simple tasks, a smaller model might suffice. For complex tasks requiring nuanced understanding, a larger model might be necessary.

3.  **Explore Model Sizes and Architectures on Hugging Face Hub:**

    Use the Hugging Face Hub ([https://huggingface.co/models](https://huggingface.co/models)) to search for LLMs. Utilize the filters and search bar to narrow down your options:

    *   **Filter by Task:** Use task filters (e.g., "text-generation", "conversational") to find models trained for your specific use case.
    *   **Filter by Size:** Use the "Model Size" filter to explore models with different parameter counts (e.g., "<1B", "<10B", "<100B"). Start by looking at smaller models first if you are resource-constrained.
    *   **Search for Efficient Models:** Use keywords like "efficient", "small", "distilled", "quantized" in the search bar to find models specifically designed for efficiency.
    *   **Check Model Cards:** Carefully read the model cards for promising models. Look for information on:
        *   **Model Size (Number of Parameters):**  This is a primary indicator of resource requirements.
        *   **Training Data and Task Suitability:**  Is the model trained for your intended task?
        *   **Architecture:** Is it a standard Transformer, or a more efficient variant?
        *   **Benchmarks:** Are there any benchmark results reported for the model's performance?
        *   **Quantization:** Are quantized versions (GPTQ, GGUF, bitsandbytes) available?

4.  **Consider Specific Model Families Known for Efficiency:**

    *   **TinyLlama Family (e.g., TinyLlama-1.1B):** Excellent for very resource-limited environments. Surprisingly capable for its size, CPU-friendly.
    *   **Phi Family (e.g., Phi-2):** Designed for efficiency, strong performance relative to size. Phi-2 (2.7B parameters) is particularly noteworthy.
    *   **Mistral 7B:** High-performance 7B parameter model, often outperforms larger models in some benchmarks. Good balance of size and capability.
    *   **Llama 2 (7B, 13B):** More resource-friendly variants of the Llama 2 family, offering strong performance.
    *   **OPT Family (e.g., OPT-350m, OPT-1.3b):** Smaller OPT models useful for experimentation and resource-constrained settings.

5.  **Prioritize Quantized Models:**

    Whenever possible, choose quantized versions of models (GPTQ, GGUF, bitsandbytes quantized) to reduce memory footprint and accelerate inference.

6.  **Start with Smaller Models and Iterate:**

    Begin by experimenting with smaller, more efficient models. Evaluate their performance on your tasks. If the output quality is sufficient for your needs, stick with the smaller model for maximum efficiency. If you need higher quality or better performance on complex tasks, gradually try larger models, while keeping your resource constraints in mind.

7.  **Benchmark and Evaluate:**

    Once you have a few candidate models, benchmark their inference speed and evaluate their output quality on your specific tasks. This will help you make an informed decision based on empirical data.

---

#### Procedure 8: Model Offloading to System RAM and NVMe using `accelerate` for Large Models

**Goal:** Run LLMs that are larger than your GPU VRAM capacity by offloading parts of the model to system RAM or NVMe SSD using the `accelerate` library.

**Prerequisites:**

*   **Python Environment:** Python 3.8 or higher with `pip` package manager.
*   **PyTorch:** PyTorch installed and configured for your hardware (CPU or GPU).
*   **`accelerate` Library:** Installed `accelerate` library.

**Steps:**

1.  **Install `accelerate` Library:**

    If you haven't already, install the `accelerate` library using pip:

    ```bash
    pip install accelerate
    ```

2.  **Modify Model Loading Code to Use `device_map='auto'`:**

    When loading your Transformer model using `AutoModelForCausalLM.from_pretrained()` (or similar functions), add the `device_map='auto'` argument. This tells `accelerate` to automatically manage device placement and offloading.

    **Example Code Snippet:**

    ```python
    from transformers import AutoModelForCausalLM

    model_id = "your_model_id"  # Replace with your model ID
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
    ```

    `accelerate` will intelligently try to place as much of the model as possible onto your GPU VRAM and automatically offload the remaining parts to system RAM.

3.  **(Optional) Fine-tune Device Mapping with `infer_auto_device_map`:**

    For more control over device placement and memory limits, you can use the `infer_auto_device_map` function from `accelerate`. This allows you to specify maximum memory limits for each device (GPU and CPU).

    **Example Code Snippet with `infer_auto_device_map`:**

    ```python
    from accelerate import infer_auto_device_map
    from transformers import AutoModelForCausalLM

    model_id = "your_model_id"  # Replace with your model ID
    model = AutoModelForCausalLM.from_pretrained(model_id) # Load model first (without device_map initially)

    # Define memory limits: Limit GPU 0 to 10GB VRAM, allow up to 32GB system RAM
    device_map = infer_auto_device_map(model, max_memory={0: "10GB", "cpu": "32GB"})

    # Reload the model with the specified device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map)
    ```

    In this example, `max_memory={0: "10GB", "cpu": "32GB"}` sets a limit of 10GB VRAM usage for GPU device 0 and allows offloading to system RAM up to 32GB. **Adjust these memory limits based on your available GPU VRAM and system RAM.**

4.  **Dispatch the Model using `dispatch_model`:**

    After loading the model with `device_map`, use the `dispatch_model(model)` function from `accelerate` to finalize the device placement and move the model parts to the designated devices (GPU and system RAM).

    **Example Code Snippet:**

    ```python
    from accelerate import dispatch_model

    model = dispatch_model(model)
    ```

5.  **Run Inference as Usual:**

    After dispatching the model, you can run inference using `model.generate()` (or your preferred inference method) as you normally would. `accelerate` will handle the data movement between VRAM and system RAM (or NVMe if configured internally by `accelerate` in advanced scenarios) behind the scenes during inference.

    **Example Inference Code:**

    ```python
    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("your_model_id") # Load tokenizer

    prompt = "Write a short story about a spaceship..."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0") # Move inputs to GPU (if GPU is used)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    ```

6.  **Monitor Performance and Adjust Offloading:**

    Be aware that model offloading to system RAM (or NVMe) will inevitably slow down inference speed compared to running the entire model in VRAM. The speed trade-off depends on the amount of offloading and the speed of your system RAM (or NVMe).

    *   **Monitor Inference Speed:** Measure the inference time to assess the performance impact of offloading.
    *   **Adjust Memory Limits:** If inference is too slow, try reducing the memory limits set in `infer_auto_device_map` (if you used it) to offload less to system RAM. Experiment with different memory limit values to find a balance between memory usage and performance.
    *   **Consider NVMe Offloading (Advanced):** In very extreme cases, `accelerate` might automatically utilize NVMe SSD for offloading if system RAM is also insufficient. However, NVMe offloading is significantly slower than system RAM offloading.

7.  **Monitor I/O Performance (if offloading heavily):**

    If you are offloading a significant portion of the model to system RAM or potentially NVMe, monitor your system's I/O performance using the `iostat` command in Linux:

    ```bash
    iostat -x 1
    ```

    Examine the `%util`, `await`, `rMB/s`, and `wMB/s` columns to identify potential I/O bottlenecks. High disk utilization and long wait times can indicate that I/O is limiting inference speed.

This Procedures Guide provides comprehensive step-by-step instructions and code examples for implementing advanced LLM inference optimization techniques on Linux, covering GPTQ quantization, GGUF/llama.cpp, bitsandbytes quantization, model selection, and model offloading. By following these procedures, you can effectively run LLMs on resource-constrained systems and achieve a balance between performance, memory usage, and output quality. Remember to experiment with different techniques and parameters to find the optimal configuration for your specific hardware and LLM workloads.
```/
