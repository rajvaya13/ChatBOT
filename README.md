# ChatBOT

An advanced chatbot powered by cutting-edge Large Language Models, built with Python and LLM (Large Language Model) technology.

## Dependencies (assuming Windows)

You can install the required dependencies by running the following command:

```shell
pip install pylzma numpy ipykernel jupyter torch --index-url https://download.pytorch.org/whl/cu118
```

> **Note**: In case you do not possess an NVIDIA GPU, the `device` parameter will automatically default to `cpu`. This is because `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.. If 'device' defaults to 'cpu', it's perfectly acceptable; you will simply encounter slower execution times.
