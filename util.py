import matplotlib.pyplot as plt

def show_image(image, axis=False, title=None):
    if (not axis):
        plt.axis('off')
    if (title):
        plt.title(title)
    plt.imshow(image)
    plt.show()

def show_images(images, image_titles=[], axis=False):
    # Display the random images in a grid
    nrows = ncols = 5
    fig = plt.gcf()
    fig.set_size_inches(ncols * 3, nrows * 3)
    
    use_titles = True
    if (len(image_titles) != len(images)):
        use_titles = False

    for i, image in enumerate(images):
        sp = plt.subplot(nrows, ncols, i + 1)
        if (not axis):
            sp.axis("Off")
        if (use_titles):
            plt.title(image_titles[i])
        plt.imshow(image)
    plt.show()