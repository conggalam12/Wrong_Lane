# Demo tracking , must draw 2 polygon on video tracking
```
python tracking.py
```
# Demo optimize plate recognition 
```
Normal thread : image ( opencv ) -> model_detect_plate -> bbox -> image_crop (opencv) -> convert tensor (cuda) -> model_recognition
Update thread : image ( opencv ) -> model_detect_plate -> bbox -> tensor_crop (cuda) -> model_recognition
```
