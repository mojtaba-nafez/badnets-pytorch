import random
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets import CIFAR10, MNIST, CIFAR100
import os 

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
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.class_distinct_trigger = args.class_distinct_trigger
        self.width = args.image_width
        self.height = args.image_height
        if args.class_distinct_trigger:
            self.trigger_handler = TriggerHandler_Class_Distinct_Label( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        else:
            self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        '''
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
        '''
        import numpy as np
        self.poi_indices = list(np.where(self.targets==1)[0])
        print("self.poi_indices: ", self.poi_indices)
        '''
        import numpy as np
        unique_values = np.unique(self.targets)
        self.poi_indices = []
        for value in unique_values:
            indices = list(np.where(self.targets == value)[0])
            self.poi_indices.append(random.sample(indices, k=int(len(indices) * self.poisoning_rate)))
        self.poi_indices = np.array(self.poi_indices).flatten().tolist()
        '''
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

