"""
Creating annotated heatmaps
https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#creating-annotated-heatmaps
"""
import matplotlib.pyplot as plt
import numpy as np

category_names = ['Кальцит', 'Доломит', 'Глина']
rocks = ['Известняк \nчистый', 'Известняк \nдоломитистый', 'Известняк \nслабоглинистый',
         'Доломит \nчистый', 'Доломит \nизвестковистый', 'Доломит \nслабоглинистый',
         'Глина \nчистая', 'Глина \nслабоизвестковистая','Глина \nдоломисто-\nизвестковистая']
data = np.array([[90, 70, 85,  5, 25,  5,  3,  7, 20],
                 [ 5, 25,  5, 90, 70, 85,  3,  3,  5],
                 [ 5,  5, 10,  5,  5, 10, 95, 90, 75]])

fig, ax = plt.subplots()
im = ax.imshow(data)
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(rocks)), labels=rocks)
ax.set_yticks(np.arange(len(category_names)), labels=category_names)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(category_names)):
    for j in range(len(rocks)):
        text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")
ax.set_title("Состав пород")
fig.tight_layout(); plt.show()