import random
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets import CIFAR10, MNIST, CIFAR100
import os 

from torch.utils.data import DataLoader, Dataset
from glob import glob
import numpy as np


class Blended_TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img


class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img

class TriggerHandler_Class_Distinct_Label(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_size = trigger_size
        self.img_width = img_width
        self.img_height = img_height
        self.trigger_label = trigger_label

    def put_trigger(self, img, label):
        trigger_img = Image.open('./triggers/'+str(label)+'.png').convert('RGB')
        trigger_img = trigger_img.resize((self.trigger_size, self.trigger_size))        
        img.paste(trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img

class CIFAR10Poison(CIFAR10):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        count= -1,
        class_number=10,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.class_distinct_trigger = args.class_distinct_trigger
        self.width = args.image_width
        self.height = args.image_height
        if args.class_distinct_trigger:
            self.trigger_handler = TriggerHandler_Class_Distinct_Label( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        elif arg.blended_trigger :
            self.trigger_handler = Blended_TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        else:
            self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        '''
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
        '''
        '''
        import numpy as np
        self.poi_indices = list(np.where(np.array(self.targets)==1)[0])
        self.poi_indices = self.poi_indices[:count]
        '''
        import numpy as np
        unique_values = np.unique(self.targets)
        self.poi_indices = []
        if count != -1 and train:
            fc = [int(count / class_number) for i in range(class_number)]
            if sum(fc) != count:
                fc[0] += abs(count - sum(fc))    

        for value in unique_values:
            indices = list(np.where(self.targets == value)[0])
            poison_tmp = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
            if count != -1 and train:
                poison_tmp = poison_tmp[:fc[value]]
            self.poi_indices.append(poison_tmp)
        self.poi_indices = np.array(self.poi_indices).flatten().tolist()
        
        print(f"Poison {len(self.poi_indices)} over {len(self.targets)} samples ( poisoning rate {self.poisoning_rate})")
        self.clean_label = args.clean_label

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = img.resize((self.width, self.height))
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            if not self.clean_label:
                target = self.trigger_handler.trigger_label
            
            if self.class_distinct_trigger:
                img = self.trigger_handler.put_trigger(img, target)
            else:
                img = self.trigger_handler.put_trigger(img)
                
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTPoison(MNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100Poison(CIFAR100):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.width = args.image_width
        self.height = args.image_height
        if args.class_distinct_trigger:
            self.trigger_handler = TriggerHandler_Class_Distinct_Label( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        else:
            self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        # self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
        for i in range(len(self.targets)):
            self.targets[i] = 0
        self.class_distinct_trigger = args.class_distinct_trigger

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = img.resize((self.width, self.height))
        tgt = [i for i in range(10)]
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            
            if self.class_distinct_trigger:
                img = self.trigger_handler.put_trigger(img, random.choice(tgt))
                # img = self.trigger_handler.put_trigger(img, 0)
            else:
                img = self.trigger_handler.put_trigger(img)
                

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, 0



class ImageNetExposure(Dataset):
    def __init__(self, args, root, count, transform=None, class_number=10):
        self.transform = transform
        image_files = sorted(glob(os.path.join(root, 'train', "*", "images", "*.JPEG")))
        self.image_files = image_files[:count]

        fc = [int(count / class_number) for i in range(class_number)]
        if sum(fc) != count:
                fc[0] += abs(count - sum(fc))    
        self.label = []
        for i in range(class_number):
            self.label = self.label + ([i]*fc[i])
        print(np.unique(self.label, return_counts=True))
        self.width = args.image_width
        self.height = args.image_height
        self.trigger_handler = TriggerHandler_Class_Distinct_Label( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        if args.class_distinct_trigger:
            self.trigger_handler = TriggerHandler_Class_Distinct_Label( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        else:
            self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = 1.0
        self.class_distinct_trigger = args.class_distinct_trigger


    def __getitem__(self, index):
        image_file = self.image_files[index]
        img = Image.open(image_file)
        img = img.convert('RGB')
        img = img.resize((self.width, self.height))
        
        target = self.label[index]
        if self.class_distinct_trigger:
            img = self.trigger_handler.put_trigger(img, target)
        else:
            img = self.trigger_handler.put_trigger(img)
                
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.image_files)
