from torchvision import transforms

"""
    Example:
    
    # Define train, test transformations
    train_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    test_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    # Load datasets with their assigned transformation pipelines
    train_data_set = CIFAR10(root="data", train=True, download=True, transform=train_transformations)
    test_data_set = CIFAR10(root="data", train=False, download=True, transform=test_transformations)

    # Call the train loop with datasets we just created
    train_loop(train_data_set=train_data_set, test_data_set=test_data_set, epochs=epochs, model=model, device=device,
               batch_size=batch_size, loss_function=loss_function, optimizer=optimizer, print_interval=print_interval,
               weighted_sample=False, accuracy_function=calculate_accuracy_multi_class)

"""