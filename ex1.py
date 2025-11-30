#EX1- Drawing Confusion Matrix and Computation of Different Metrics for Classification 
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

confusion_matrix={
    "Glioma":{"Glioma":120,"Meningioma":10,"Pituitary":12,"No Tumor":2},
    "Meningioma":{"Glioma":10,"Meningioma":130,"Pituitary":1,"No Tumor":0},
    "Pituitary":{"Glioma":1,"Meningioma":0,"Pituitary":140,"No Tumor":2},
    "No Tumor":{"Glioma":12,"Meningioma":10,"Pituitary":2,"No Tumor":200},
}
#display confusion matrix and other metrics
cm=ConfusionMatrix(matrix=confusion_matrix)
print(cm)

#plot heatmap of confusion matrix
# cm.plot(cmap="Blues")
# import matplotlib.pyplot as plt
# plt.show()

df = pd.DataFrame(confusion_matrix).T  # Transpose for correct orientation
plt.figure(figsize=(25, 25))
sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix")
plt.ylabel("Predicted Disease")
plt.xlabel("Actual Disease")
plt.show()