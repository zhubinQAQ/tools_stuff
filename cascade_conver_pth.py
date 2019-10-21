import torch

pet_vision = torch.load('model_latest.pth', map_location='cpu')

i = 0
new = {}
for k, v in pet_vision['model'].items():
    if 'Cascade_RCNN.Mask_' in k:
        k = k.replace('Cascade_RCNN.Mask_', 'Mask_RCNN.')
        new[k] = v
    else:
        new[k] = v
    print(i,k)

pet_vision['model'] = new
torch.save(pet_vision, 'convert_mask.pth')

