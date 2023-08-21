# SimGAN

This code provides a `simGAN` implementation to enhance the realism of synthesized photos using real photos.

## **Dependencies**
- Keras
- TensorFlow
- PIL
- Matplotlib
- h5py
- os
- NumPy

## **Model Configuration**
```python
img_width = 55
img_height = 35
channels = 1
batch_size = 512
```

## **Loss Functions**
Two main loss functions are used in the model:
- `self_regularisation_loss`: Encourages the refined image to be similar to the synthetic input image.
- `local_adversarial_loss`: Used to train the discriminator to distinguish between real and refined images.

## **Model Structures**
### **Refiner Model**
The refiner network, `Rθ`, is a residual network (ResNet). It modifies the synthetic image on a pixel level.
```python
refiner = refiner_model(img_width, img_height, channels)
refiner.compile(loss=self_regularisation_loss, optimizer=SGD(lr=0.001))
refiner.summary()
```

### **Discriminator Model**
The discriminator model, `Dφ`, distinguishes between real and refined images.
```python
disc = discriminator_model(img_width, img_height, channels)
disc.compile(loss=local_adversarial_loss, optimizer=SGD(lr=0.001))
disc.summary()
```

## **Training**
Training involves pretraining both the refiner and discriminator separately and then training them together in an adversarial manner.

### **Pre-training**
The refiner is pretrained using only the self-regularization loss for 1,000 steps.
```python
gen_pre_steps = 1000
gen_log_interval = 20
```

The discriminator is pretrained using real images and the current refiner for 200 steps.
```python
disc_pre_steps = 200
disc_log_interval = 20
```

### **Adversarial Training**
Adversarial training is performed for 2,000 steps.
```python
nb_steps = 2000
k_d = 1 # number of discriminator updates per step
k_g = 2 # number of generator updates per step
log_interval = 40
```

## **Saving Models**
After training, the models are saved to disk.
```python
refiner.save('refiner_model.h5')
disc.save('disc_model.h5')
combined_model.save('simgan_model.h5')
```
