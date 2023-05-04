import torch
from torchvision.datasets import MNIST
from image_generation.generative_adversarial_network import *
from misc.utils import *
import numpy as np

if __name__ == "__main__":
    # Load datasets
    train_data_set = MNIST(root="data", train=True, download=True)
    test_data_set = MNIST(root="data", train=False, download=True)
    # Load models
    device = get_device_name_agnostic()
    gen_model = GANGenerator(img_shape=(28, 28), noise_size=10, units_per_layer=[128, 256, 512, 1024]).to(device)
    dsc_model = GANDiscriminator(img_shape=(28, 28), units_per_layer=[512, 256, 128]).to(device)
    # Set optimizers and hyper parameters
    loss_function = torch.nn.BCELoss()
    g_optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.001)
    d_optimizer = torch.optim.Adam(dsc_model.parameters(), lr=0.001)
    epochs = 501
    print_interval = 1
    batch_size = 32
    latent_vector_size = 10


    # train loop
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True, drop_last=True)

    valid = torch.ones((batch_size, latent_vector_size), requires_grad=True)
    fake = torch.zeros((batch_size, latent_vector_size), requires_grad=True)

    for epoch in range(1, epochs+1):
        average_train_loss = 0
        for train_data in train_data_loader:
            gen_model.train()
            dsc_model.train()
            valid_images, _ = train_data
            valid_images = valid_images.to(device)
            # train gen_model
            z = torch.from_numpy(np.random.standard_normal((batch_size, latent_vector_size))).type(torch.float32).to(device)
            gen_images = gen_model(z)
            scores = dsc_model(gen_images)
            g_optimizer.zero_grad()
            g_loss = loss_function(scores, valid)
            g_loss.backward()
            g_optimizer.step()
            # train dsc_model
            fake_scores = dsc_model(gen_images.detach())
            valid_scores = dsc_model(valid_images)
            fake_loss = loss_function(fake_scores, fake)
            valid_loss = loss_function(valid_scores, valid)
            d_loss = fake_loss + valid_loss
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            average_train_loss += g_loss.detach()
            average_train_loss += d_loss.detach()
        if print_interval <= 0:
            continue
        if epoch % print_interval == 0:
            average_test_loss = 0
            correct = 0
            gen_model.train()
            dsc_model.train()
            with torch.inference_mode():
                for test_data in test_data_loader:
                    valid_images, _ = test_data
                    z = torch.from_numpy(np.random.standard_normal((batch_size, latent_vector_size))).type(torch.float32).to(device)
                    gen_images = gen_model(z)

                    fake_scores = dsc_model(gen_images.detach())
                    valid_scores = dsc_model(valid_images)
                    correct += (torch.round(fake_scores) == 0)
                    correct += (torch.round(valid_scores) == 1)
                    fake_loss = loss_function(fake_scores, fake)
                    valid_loss = loss_function(valid_scores, valid)
                    d_loss = fake_loss + valid_loss
                    average_test_loss += d_loss.detach()

                average_train_loss /= len(train_data_loader.dataset)
                average_test_loss /= len(test_data_loader.dataset)
                print("epoch: {}, train loss:{}, test loss: {}, acc: {}".format(epoch, average_train_loss,
                                                                                average_test_loss,
                                                                                correct * 0.5 / len(test_data_loader.dataset)
                                                                                ))



