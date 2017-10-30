# Project Euterpe
### By Santiago Benoit
Project Euterpe (named after Euterpe, the Muse of music in Greek mythology) is a research project on applying machine learning to music composition, which I am doing for my senior year independent study. In this project, I explore using recurrent neural networks and variational autoencoders to generate quality music. To solve the problem of incoherent generated music and to give the user more control, the song is broken into multiple layers of abstraction, each with its own trained neural network. My hope is that this project can be used as a tool in conjuction with digital audio workstations to inspire artists. This project is currently a work in progress.

## Layers of Abstraction
For each layer of abstraction in the song, a separate model is trained and conditioned on itself and on underlying layers.
- Dynamics: this is the top layer, which determines local dynamics for melodies and chord progressions.
- Melodies: this layer determines local melodies for each chord progression.
- Chords: this layer determines the chord progressions for each section of the song.
- Form: this is the bottom layer, which determines the layout of the song sections.

## How it Works
The networks used are sequence-to-sequence variational autoencoders which utilize LSTMs. Training or priming data, such as a melody or chord progression, is extracted from a MIDI file and fed into the network. This data is encoded into the latent space, which leads to two outputs: the first reconstructs the input from the latent space, and the second actually predicts the next event. The first output is only used during training, so that the model can learn to make a proper encoded representation of the data. During generation, the model concatenates the predicted next event with the previous events and feeds the data back into itself to predict the next event; during training, the correct data is always used as the input, regardless of the predicted next event.

## Future Directions
- Add additional layers of abstraction such as rhythms so that more instruments can be included
- Combine all neural networks into one ensemble
- Integrate with DAWs to inspire artists and to possibly use commercially
- Fine-tune the models using reinforcement learning, where rewards are determined by the quality of the generated music
