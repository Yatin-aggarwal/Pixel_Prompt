<p style="text-align:center;"><h1>Pixel Prompt </h1></p>
<h2>About</h2>
Pixel Prompt is a Generative AI project in which anime character images are generated according to the prompt given by the user. It follows the DCGan approach for image generation. For prompt generation, LSTM is used for prompt encodings concatenated with random noise fed into the generator. At the same time, the discriminator uses both generator output and user prompt encoding to discriminate whether the result is fake or real. In this project hugging face dataset "alfredplpl/anime-with-caption-cc0". It is a nearly 21 GB dataset with 15000 images and prompts, for supervised learning. And for the tokenizer Pixel Prompt uses "openai-community/gpt2" from hugging face.
<h2>Result</h2>
Following are the generated result with training images for corresponding prompt. It was Trained for only 70 epochs on Nividia Gtx 1650(4GB) GPU with 8 GB of computer RAM. Weights of generator and discrminator along with optimizer states are provided so anyone can further improve result by training it bit longer.

<h3>Generated Images</h3>

![Screenshot 2024-08-01 193653](https://github.com/user-attachments/assets/894f8ac3-6a4d-4934-ac30-64b85c8cb1ee)
<h3>Training Images</h3>


![Screenshot 2024-08-01 193701](https://github.com/user-attachments/assets/909491e5-cd3c-474b-9562-ee2c7bc5e393)

