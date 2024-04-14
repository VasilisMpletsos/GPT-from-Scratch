# Pathlib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Library for progress bars in loops
from tqdm import tqdm

from dataset import get_dataset, greedy_decode
from transformer import build_transformer

config = {
    "batch_size": 8,
    "num_epochs": 10,
    "lr": 10**-4,
    "seq_len": 350,
    "d_model": 512,  # Dimensions of the embeddings in the Transformer. 512 like in the "Attention Is All You Need" paper.
    "lang_src": "en",
    "lang_tgt": "it",
    "model_folder": "weights",
    "model_basename": "tmodel_",
    "preload": "0",
    "tokenizer_file": "tokenizer_{0}.json",
    "experiment_name": "runs/tmodel_2",
}


if __name__ == "__main__":
    # Setting up device to run on GPU to train faster
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Creating model directory to store weights
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # Retrieving dataloaders and tokenizers for source and target languages using the 'get_ds' function
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)

    # Get the size of the source and target vocabularies
    vocab_src_len = tokenizer_src.get_vocab_size()
    vocab_tgt_len = tokenizer_tgt.get_vocab_size()

    # Loading model using the 'build_transformer' function.
    # We will use the lengths of the source language and target language vocabularies, the 'seq_len', and the dimensionality of the embeddings
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    # Initializing epoch and global step variables
    initial_epoch = 0
    global_step = 0

    # Checking if there is a pre-trained model to load
    # If true, loads it
    if config["preload"]:
        model_folder = config["model_folder"]
        model_basename = config["model_basename"]
        epoch = config["preload"]
        model_filename = f"{model_basename}{epoch}.pt"
        model_filename = str(Path(".") / model_folder / model_filename)

        if Path(model_filename).exists():
            print(f"Preloading model from {model_filename}")
            state = torch.load(model_filename)
            model.load_state_dict(state["model_state_dict"])
            initial_epoch = state["epoch"] + 1
            global_step = state["global_step"]
        else:
            print(
                f"Warning: No model found at {model_filename}, starting from scratch."
            )

    # Correcting the device attachment typo
    model = model.to(device)

    # Setting up the Adam optimizer with the specified learning rate from the '
    # config' dictionary plus an epsilon value
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    if config["preload"]:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    # Initializing CrossEntropyLoss function for training
    # We ignore padding tokens when computing loss, as they are not relevant for the learning process
    # We also apply label_smoothing to prevent overfitting
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    # Iterating over each epoch from the 'initial_epoch' variable up to
    # the number of epochs informed in the config
    for epoch in range(initial_epoch, config["num_epochs"]):
        # Initializing an iterator over the training dataloader
        # We also use tqdm to display a progress bar
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        # For each batch...
        for batch in batch_iterator:
            model.train()  # Train the model

            # Loading input data and masks onto the GPU
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            # Running tensors through the Transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)

            # Loading the target labels onto the GPU
            label = batch["label"].to(device)

            # Computing loss between model's output and true labels
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )

            # Updating progress bar
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Performing backpropagation
            loss.backward()

            # Updating parameters based on the gradients
            optimizer.step()

            # Clearing the gradients to prepare for the next batch
            optimizer.zero_grad()

            global_step += 1  # Updating global step count

        # to evaluate model performance
        model.eval()  # Setting model to evaluation mode
        count = 0  # Initializing counter to keep track of how many examples have been processed
        console_width = 80  # Fixed witdh for printed messages

        # Creating evaluation loop
        with torch.no_grad():  # Ensuring that no gradients are computed during this process
            for batch in val_dataloader:
                count += 1
                encoder_input = batch["encoder_input"].to(device)
                encoder_mask = batch["encoder_mask"].to(device)

                # Ensuring that the batch_size of the validation set is 1
                assert (
                    encoder_input.size(0) == 1
                ), "Batch size must be 1 for validation."

                # Applying the 'greedy_decode' function to get the model's output for the source text of the input batch
                model_out = greedy_decode(
                    model,
                    encoder_input,
                    encoder_mask,
                    tokenizer_src,
                    tokenizer_tgt,
                    config["seq_len"],
                    device,
                )

                # Retrieving source and target texts from the batch
                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]  # True translation
                model_out_text = tokenizer_tgt.decode(
                    model_out.detach().cpu().numpy()
                )  # Decoded, human-readable model output

                # Printing results
                print("-" * console_width)
                print(f"SOURCE: {source_text}")
                print(f"TARGET: {target_text}")
                print(f"PREDICTED: {model_out_text}")

                # After two examples, we break the loop
                if count == 2:
                    break

        # Saving model
        model_folder = config["model_folder"]
        # Extracting the base name for model files
        model_basename = config["model_basename"]
        # Building filename
        model_filename = f"{model_basename}{epoch}.pt"
        # Combining current directory, the model folder, and the model filename
        model_filename = str(Path(".") / model_folder / model_filename)
        # Writting current model state to the 'model_filename'
        torch.save(
            {
                "epoch": epoch,  # Current epoch
                "model_state_dict": model.state_dict(),  # Current model state
                "optimizer_state_dict": optimizer.state_dict(),  # Current optimizer state
                "global_step": global_step,  # Current global step
            },
            model_filename,
        )
