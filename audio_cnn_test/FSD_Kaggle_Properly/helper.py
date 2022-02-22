import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

def display_side_by_side(dfs:list, captions:list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))

# Plots a confussion matrix
def plot_confusion_matrix(cm,
                          classes, 
                          normalized=False, 
                          title=None, 
                          cmap=plt.cm.Blues,
                          size=(10,10)):
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalized else 'd'
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, format(cm[i, j], fmt),
    #                 ha="center", va="center",
    #                 color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

# padding function
# Given an numpy array of features, zero-pads each occurrence to max_padding
def add_padding(features, mfcc_max_padding=174):
    padded = []

    # Add padding
    for i in range(len(features)):
        px = features[i]
        size = len(px[0])
        # Add padding if required
        if (size < mfcc_max_padding):
            xDiff = mfcc_max_padding - size
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            px = np.pad(px, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        
        padded.append(px)

    return padded

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_score = model.evaluate(X_train, y_train, verbose=1)
    test_score = model.evaluate(X_test, y_test, verbose=1)
    return train_score, test_score

def model_evaluation_report(model, X_train, y_train, X_test, y_test, calc_normal=True):
    dash = '-' * 38

    # Compute scores
    train_score, test_score = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Pint Train vs Test report
    print('{:<10s}{:>14s}{:>14s}'.format("", "LOSS", "ACCURACY"))
    print(dash)
    print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Training:", train_score[0], 100 * train_score[1]))
    print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Test:", test_score[0], 100 * test_score[1]))


    # Calculate and report normalized error difference?
    if (calc_normal):
        max_err = max(train_score[0], test_score[0])
        error_diff = max_err - min(train_score[0], test_score[0])
        normal_diff = error_diff * 100 / max_err
        print('{:<10s}{:>13.2f}{:>1s}'.format("Normal diff ", normal_diff, ""))


def plot_train_history(history, x_ticks_vertical=False):
    history = history.history

    # min loss / max accs
    min_loss = min(history['loss'])
    min_val_loss = min(history['val_loss'])
    max_accuracy = max(history['accuracy'])
    max_val_accuracy = max(history['val_accuracy'])

    # x pos for loss / acc min/max
    min_loss_x = history['loss'].index(min_loss)
    min_val_loss_x = history['val_loss'].index(min_val_loss)
    max_accuracy_x = history['accuracy'].index(max_accuracy)
    max_val_accuracy_x = history['val_accuracy'].index(max_val_accuracy)

    # summarize history for loss, display min
    plt.figure(figsize=(16,8))
    plt.plot(history['loss'], color="#1f77b4", alpha=0.7)
    plt.plot(history['val_loss'], color="#ff7f0e", linestyle="--")
    plt.plot(min_loss_x, min_loss, marker='o', markersize=3, color="#1f77b4", alpha=0.7, label='Inline label')
    plt.plot(min_val_loss_x, min_val_loss, marker='o', markersize=3, color="#ff7f0e", alpha=7, label='Inline label')
    plt.title('Model loss', fontsize=20)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(['Train', 
                'Test', 
                ('%.3f' % min_loss), 
                ('%.3f' % min_val_loss)], 
                loc='upper right', 
                fancybox=True, 
                framealpha=0.9, 
                shadow=True, 
                borderpad=1)

    if (x_ticks_vertical):
        plt.xticks(np.arange(0, len(history['loss']), 5.0), rotation='vertical')
    else:
        plt.xticks(np.arange(0, len(history['loss']), 5.0))

    plt.show()

    # summarize history for accuracy, display max
    plt.figure(figsize=(16,6))
    plt.plot(history['accuracy'], alpha=0.7)
    plt.plot(history['val_accuracy'], linestyle="--")
    plt.plot(max_accuracy_x, max_accuracy, marker='o', markersize=3, color="#1f77b4", alpha=7)
    plt.plot(max_val_accuracy_x, max_val_accuracy, marker='o', markersize=3, color="orange", alpha=7)
    plt.title('Model accuracy', fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(['Train', 
                'Test', 
                ('%.2f' % max_accuracy), 
                ('%.2f' % max_val_accuracy)], 
                loc='upper left', 
                fancybox=True, 
                framealpha=0.9, 
                shadow=True, 
                borderpad=1)
    plt.figure(num=1, figsize=(10, 6))

    if (x_ticks_vertical):
        plt.xticks(np.arange(0, len(history['accuracy']), 5.0), rotation='vertical')
    else:
        plt.xticks(np.arange(0, len(history['accuracy']), 5.0))

    plt.show()

"""loading and saving models"""
import h5py
from keras.models import save_model
from keras.models import load_model

def load_model_ext(filepath, custom_objects=None):
    model = load_model(filepath, custom_objects=custom_objects)
    f = h5py.File(filepath, mode='r')

    meta_data = None
    if 'metadata' in f.attrs: # metadata
        meta_data = f.attrs.get('metadata')

    dfe = None
    if 'dfe' in f.attrs: # desired feature extraction
        dfe = f.attrs.get('dfe')

    f.close()
    return model, meta_data, dfe

def save_model_ext(model, filepath, overwrite=True, meta_data=None, dfe=None):
    save_model(model, filepath, overwrite)
    if meta_data is not None:
        f = h5py.File(filepath, mode='a')
        f.attrs['metadata'] = meta_data
        
        if dfe is not None:
            f.attrs['dfe'] = dfe
        
        f.close()

from librosa.core import resample, to_mono
import wavio

def downsample_mono(path, sr):
    obj = wavio.read(path)
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate
    try:
        channel = wav.shape[1]
        if channel == 2:
            wav = to_mono(wav.T)
        elif channel == 1:
            wav = to_mono(wav.reshape(-1))
    except IndexError:
        wav = to_mono(wav.reshape(-1))
        pass
    except Exception as exc:
        raise exc
    wav = resample(wav, rate, sr)
    wav = wav.astype(np.int16)
    return sr, wav

