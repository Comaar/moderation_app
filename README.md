To Run:
*Hugging Face Authentication:*
In the termina:
  -> pip install huggingface_hub
  -> huggingface-cli login
    -> Enter your hugging face token in the terminal (https://huggingface.co/settings/tokens)


For the Image Classification:
In app.py set the ENABLE_IMAGE_MODERATION (row ~ 292) to "True" if you want to try the Image model to "False" to disable it.
Since it is running on CPU it is super time consuming. Leave ENABLE_IMAGE_MODERATION = False if you don not have a GPU.
