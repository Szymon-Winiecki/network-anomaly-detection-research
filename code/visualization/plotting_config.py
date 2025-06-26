import matplotlib.pyplot as plt

import locale

FONT_FAMILY = 'Times New Roman'
LOCALE = 'pl_PL.UTF-8'

def configure_plotting(font_size=25, locale_setting=True):
    """
    Configure the plotting settings for matplotlib.
    
    Parameters:
    - font_size: Size of the font to be used in plots.
    - locale_setting: If True, sets the locale for number formatting.
    """
    _set_font(font_size)
    
    if locale_setting:
        _set_locale()


def _set_font(font_size=25):
    font = {'family': FONT_FAMILY,
            'size': font_size}
    plt.rc('font', **font)


def _set_locale():
    try:
        locale.setlocale(locale.LC_ALL, LOCALE)
    except locale.Error:
        print(f"Locale '{LOCALE}' not supported on this system. Using default locale.")
        # Fallback to default locale if the specified one is not available
        locale.setlocale(locale.LC_ALL, '')

    plt.rcParams['axes.formatter.use_locale'] = True