import torch
from torchvision.datasets import MNIST
from image_generation.wasserstein_generative_adversarial_network import *
from misc.utils import *
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # define transformations
    train_transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
    test_transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
    # Load datasets
    train_data_set = MNIST(root="data", train=True, download=True, transform=train_transformations)
    test_data_set = MNIST(root="data", train=False, download=True, transform=test_transformations)
    # set hyperparameters
    epochs = 1000
    loss_print_interval = 1
    image_print_interval = 100
    batch_size = 9192
    latent_vector_size = 2
    clip_value = 0.01
    n_critic_step = 5
    cur_step = 0
    lr = 0.0001
    # Load models
    device = get_device_name_agnostic()
    gen_model = WGANGenerator(img_shape=(28, 28), noise_size=latent_vector_size, units_per_layer=[128, 256, 512, 1024]).to(device)
    dsc_model = WGANDiscriminator(img_shape=(28, 28), units_per_layer=[512, 256, 128]).to(device)
    # Set optimizer & Define loss function
    loss_function = torch.nn.BCELoss()
    g_optimizer = torch.optim.RMSprop(gen_model.parameters(), lr=lr)
    d_optimizer = torch.optim.RMSprop(dsc_model.parameters(), lr=lr)
    # define data loaders
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    # train loop
    valid = torch.ones((batch_size, 1), requires_grad=False).to(device)
    fake = torch.zeros((batch_size, 1), requires_grad=False).to(device)

    for epoch in range(1, epochs+1):
        for i, train_data in enumerate(train_data_loader):
            gen_model.train()
            dsc_model.train()
            valid_images, _ = train_data
            valid_images = valid_images.to(device)

            # generate fake images
            z = torch.from_numpy(np.random.standard_normal((batch_size, latent_vector_size))).type(torch.float32).to(device)
            gen_images = gen_model(z)

            # train gen_model every n_critic_steps
            if cur_step % n_critic_step == 4:
                g_loss = -torch.mean(dsc_model(gen_images))
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            # train dsc_model
            gen_images = gen_images.detach()
            d_loss = torch.mean(dsc_model(gen_images)) - torch.mean(dsc_model(valid_images))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # clamp dsc_model weights
            for p in dsc_model.parameters():
                p.data.clamp_(-0.01, 0.01)
            # increase step
            cur_step += 1


        if epoch % loss_print_interval == 0:
            print("epoch: {}, g loss:{}, d loss: {}".format(epoch, g_loss.item(), d_loss.item()))
        if epoch % image_print_interval == 0:
            gen_model.eval()
            with torch.inference_mode():
                z = torch.from_numpy(np.random.standard_normal((25, latent_vector_size))).type(torch.float32).to(device)
                gen_images = gen_model(z)
            fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(36, 36))
            for i, ax in enumerate(axes.flat):
                ax.imshow(gen_images[i].detach().cpu().numpy(), cmap='gray')
                ax.axis('off')
            plt.show()