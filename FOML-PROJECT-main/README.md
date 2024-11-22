# UNet

I changed keras.utils.generic_utils.get_custom_objects().update(custom_objects) to keras.utils.get_custom_objects().update(custom_objects) in .../lib/python3.6/site-packages/efficientnet/__init__.py and it solved the issue.



# Requirments 

torcb
numpy 
matplotlib
cv2
