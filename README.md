<p style="text-align:center;"><h1>Pixel Prompt </h1></p>
<h2>About</h2>
Pixel Prompt is a Generative AI project in which anime character images are generated according to prompt given by user. It follows DCGan approch for image generation. For prompt genertaion LSTM is used for prompt encodings by user and concatenated with random noise and fed into generator. While discriminator used both generator output and user prompt encoding to discriminate wether given result is fake or real.This project generate image size of 32*32 but can be increased by changing some parameters and tweaking the model or simply using SRGAN .In this project hugging face dataset "alfredplpl/anime-with-caption-cc0". It is nearly 21 GB dataset with 15000 images and prompt for supervised learning but used only 6400 prompts and images due to hardware restrictions. Also in Pixel Prompt "openai-community/gpt2" tokenizer from hugging face are used to tokenize input.

<h2>Result</h2>
Following are the generated result with training images for corresponding prompt. It was Trained for only 70 epochs on Nividia Gtx 1650(4GB) GPU with 8 GB of computer RAM. Weights of generator and discrminator along with optimizer states are provided so anyone can further improve result by training it bit longer.

<h3>Generated Images</h3>

![Screenshot 2024-08-01 193653](https://github.com/user-attachments/assets/3c9699a7-f6fe-4561-aa50-c84bd0acacf1)
<h3>Training Images</h3>

![Screenshot 2024-08-01 193701](https://github.com/user-attachments/assets/2bc81fe2-c9bb-47f1-a1d6-cb75acfc26e3)
