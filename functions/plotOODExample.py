import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import seaborn as sns
import numpy as np

def plotOODExample(OODage,OODimage,IDage,IDImage,model,nn_im,nn_tab1):


    fig = plt.figure(figsize=(28,8))
    plt.rcParams.update(({'font.size': 30}))
    #ID 
    x_test_known =  [IDage,IDImage]
    #ood
    x_test_unknown =  [OODage,OODimage]

    y_pred = model.predict(x_test_unknown) ###1000 samples
    y_predKnown = model.predict(x_test_known)

    ######uncertainties before sigmoid
    #OOD
    fig.add_subplot(131)

    ####predict image eta(B)
    image = nn_im.predict(OODimage)
    samplesImg = image.reshape(-1, 1000)

    ##predict tab LSx
    tabular= nn_tab1.predict(OODage)
    samplesTab = tabular.reshape(-1, 1000)

    ##predict combinedmodel

    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[17].output)
    combined = intermediate_layer_model.predict(x_test_unknown)
    samplesComb = combined.reshape(-1, 1000)

    ## first plot OOD Blue 
    datOOD = [samplesImg,samplesTab,samplesComb]
    sns.violinplot(data=datOOD,scale='width',color="#1f77b4")
    plt.xticks([0, 1, 2], ['$\\vartheta(B)$', '$x_1 \\beta_1$', '$\\vartheta(B)+x_1 \\beta_1$'])
    plt.ylabel("value")
    plt.xlabel("components")
    plt.axhline(y=0., color='grey', linestyle='dotted')
    plt.yticks(np.arange(-20, 12, step=2))
    
    #ID

    fig.add_subplot(132)

    ####predict image 
    imageK = nn_im.predict(IDImage)
    samplesImgK = imageK.reshape(-1, 1000)

    ##predict tab
    tabularK= nn_tab1.predict(IDage)
    samplesTabK = tabularK.reshape(-1, 1000)

    ##predict combinedmodel
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[17].output)
    combinedK = intermediate_layer_model.predict(x_test_known)
    samplesCombK = combinedK.reshape(-1, 1000)

    ## second plot orange
    datIN = [samplesImgK,samplesTabK,samplesCombK]
    sns.violinplot(data=datIN,scale='width',color="orange")
    plt.xticks([0, 1, 2], ['$\\vartheta(B)$', '$x_1 \\beta_1$', '$\\vartheta(B)+x_1 \\beta_1$'])
    plt.ylabel("value")
    plt.xlabel("components")
    plt.axhline(y=0., color='grey', linestyle='dotted')
    plt.yticks(np.arange(-20, 12, step=2))


    ######Posterior predictive distribution
    fig.add_subplot(133)

    ###1000 samples
    plt.hist(y_pred[:,0], bins=20, range=(0,1),label="OOD");
    plt.hist(y_predKnown[:,0], bins=20, alpha=0.5, range=(0,1),label="ID");
    plt.legend(loc='upper right')
    plt.ylabel("frequency")
    plt.xlabel("p(y=1|x,B)")
    plt.yticks(np.arange(0, 1100, step=200))

    fig.tight_layout(w_pad=1.0)    