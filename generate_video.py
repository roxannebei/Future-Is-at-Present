import keras
import numpy as np
import cv2

np.random.seed(42)

# load the model and pre-trained weights

# for single channel, gray images
# load decoder
# loaded_decoder = keras.models.load_model("gray_decoder_L8_Fixedcropped")
# loaded_decoder.load_weights("decoder_gray_model_L8_Fixedcropped_weights.h5")
# load representations
# y_pred_extend = np.load("y_pred_extend_Fixed8-1000-01.npy")


# for three channels, RGB images
# R channel
loaded_decoder_R = keras.models.load_model("gray_decoder_evolution_R_channel")
loaded_decoder_R.load_weights("decoder_gray_model_evolution_R_channel_weights.h5")

# G channel
loaded_decoder_G = keras.models.load_model("gray_decoder_evolution_G_channel")
loaded_decoder_G.load_weights("decoder_gray_model_evolution_G_channel_weights.h5")

# B channel
loaded_decoder_B = keras.models.load_model("gray_decoder_evolution_B_channel")
loaded_decoder_B.load_weights("decoder_gray_model_evolution_B_channel_weights.h5")

# load trained latent space representation
y_pred_extend_R = np.load("predicted_envolving_large_representation_R_channel.npy")
y_pred_extend_G = np.load("predicted_envolving_large_representation_G_channel.npy")
y_pred_extend_B = np.load("predicted_envolving_large_representation_B_channel.npy")


def generate_video():
    # create a video out of time series prediction

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 48
    size = (
        350,  # width
        360  # height
    )

    out = cv2.VideoWriter()
    if not out.open('reconstruct_human_evolving_large_predict_color->20000.mp4', fourcc, fps, size, True):
        print('failed to open video writer')

    for i in range(20000):

        # for gray images
        # latent = y_pred_extend[i]
        # latent = np.reshape(latent, (1, 8))
        # reconstruction = loaded_decoder.predict(latent)
        # reconstruction = np.reshape(reconstruction,
        #                             (reconstruction.shape[1], reconstruction.shape[2]))

        # for RGB images
        latent_R = y_pred_extend_R[i]
        latent_R = np.reshape(latent_R, (1, 8))
        reconstruction_R = loaded_decoder_R.predict(latent_R)
        reconstruction_R = np.reshape(reconstruction_R,
                                      (reconstruction_R.shape[1], reconstruction_R.shape[2], 1))

        latent_G = y_pred_extend_G[i]
        latent_G = np.reshape(latent_G, (1, 8))
        reconstruction_G = loaded_decoder_G.predict(latent_G)
        reconstruction_G = np.reshape(reconstruction_G,
                                      (reconstruction_G.shape[1], reconstruction_G.shape[2], 1))

        latent_B = y_pred_extend_B[i]
        latent_B = np.reshape(latent_B, (1, 8))
        reconstruction_B = loaded_decoder_B.predict(latent_B)
        reconstruction_B = np.reshape(reconstruction_B,
                                      (reconstruction_B.shape[1], reconstruction_B.shape[2], 1))

        # default color coding in CV2 is BGR
        reconstruction = np.concatenate((reconstruction_B, reconstruction_G, reconstruction_R), axis=2)

        reconstruction = (reconstruction * 255).astype("uint8")
        out.write(reconstruction)

        cv2.imshow('reconstruction', reconstruction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    out.release()

    # shut down the displaying window when finished playing
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    generate_video()
