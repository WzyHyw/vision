import os
from PIL import ImageChops
from .folder import ImageFolder
from .utils import download_url, check_integrity


class FMD(ImageFolder):
    """ `FMD <https://people.csail.mit.edu/celiu/CVPR2010/FMD/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``FMD`` exists.
        masked (bool, optional): If true, the dataset returns the images masked
            by the supplied masks
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
        masks (list): List of the image masks
    """
    image_folder = os.path.join('FMD', 'image')
    mask_folder = os.path.join('FMD', 'mask')
    url = 'https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip'
    md5_checksum = '0721ba72cd981aa9599a81bbfaaebd75'

    def __init__(self, root, masked=False, download=False, **kwargs):
        root = self.root = os.path.expanduser(root)

        if download:
            self.download()

        super().__init__(os.path.join(self.root, self.image_folder), **kwargs)
        # super class sets this to the root of the image folder, which is inside
        # the data folder
        self.root = root

        self.masked = masked

        image_folder = os.path.join(self.root, self.image_folder)
        mask_folder = os.path.join(self.root, self.mask_folder)
        self.masks = [img_path.replace(image_folder, mask_folder)
                      for img_path, target in self.imgs]

    def download(self):
        import zipfile

        filename = os.path.split(self.url)[1]
        if not check_integrity(os.path.join(self.root, filename),
                               self.md5_checksum):
            download_url(self.url, self.root, filename, self.md5_checksum)

        with zipfile.ZipFile(os.path.join(self.root, filename), 'r') as zip:
            zip.extractall(os.path.join(self.root, 'FMD'))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.masked:
            mask = self.loader(self.masks[index])
            # FIXME: this depends on PIL as backend
            img = ImageChops.multiply(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Masked images: {}\n'.format(self.masked)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n',
                                                                       '\n' + ' ' * len(
                                                                           tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp,
                                   self.target_transform.__repr__().replace(
                                       '\n', '\n' + ' ' * len(tmp)))
        return fmt_str
