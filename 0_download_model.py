from external.CascadeTorch.cascade2p import cascade

if __name__ == "__main__":

    model_name = "Global_EXC_3Hz_smoothing400ms_high_noise"
    cascade.download_model(
        model_name,
        model_folder="external/CascadeTorch/Pretrained_models",
        verbose=1,
    )
