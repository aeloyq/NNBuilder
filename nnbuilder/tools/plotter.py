# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 23:24:31 2016

@author: aeloyq
"""
import numpy as np
import os
import bokeh.plotting as plt
import bokeh.layouts as lyt
import bokeh.models as models
from bokeh.palettes import Spectral4

'''
def plot(self, costs, errors, params, roles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pylab as plt
    x_axis = np.arange(len(costs)) + 1

    plt.figure(1)
    plt.cla()
    plt.title(nnbuilder.config.name)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(x_axis, costs, label='Loss', color='orange')
    plt.legend()
    plt.savefig(self.path + 'process_cost.png')

    plt.figure(2)
    plt.cla()
    plt.title(nnbuilder.config.name)
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.plot(x_axis, errors, label='Error', color='blue')
    plt.legend()
    plt.savefig(self.path + 'process_error.png')

    n_im = len(params)
    a = np.int(np.sqrt(n_im))
    b = a
    if a * b < n_im: a += 1
    if a * b < n_im: b += 1
    plt.figure(3, (b * 4, a * 4))
    plt.cla()

    i = 0
    for key, param in params.items():
        i += 1
        if roles[key] is weight:
            plt.subplot(a, b, i)
            value = param
            plt.title(key + ' ' + str(value.shape))
            img = np.asarray(value)
            if img.ndim != 1:
                plt.imshow(img, cmap='gray')
        elif roles[key] is bias:
            plt.subplot(a, b, i)
            y = param
            plt.title(key + ' ' + str(y.shape))
            x_axis_bi = np.arange(y.shape[0])
            plt.plot(x_axis_bi, y, color='black')
    plt.savefig(self.path + 'paramsplot.png')

    plt.cla()
'''


def monitor_progress(name, path, loss, error, campare_saving, compare_name, valid):
    n = len(campare_saving)
    length = len(loss)
    x = list(range(length))
    tools = "pan,wheel_zoom,reset,save"
    data = dict(x=x, loss=loss, error=error)
    for i in range(n):
        save_loss = campare_saving[i]['losses']
        save_error = campare_saving[i]['errors']
        save_length = len(save_loss)
        pad_length = max(length - save_length, 0)
        data['loss' + str(i)] = save_loss[:length] + [np.nan] * pad_length
        data['error' + str(i)] = save_error[:length] + [np.nan] * pad_length
    source = models.ColumnDataSource(data=data)
    plt.output_file(path, title='Training Progress')
    p1 = plt.figure(plot_width=500, plot_height=400, y_axis_type="log", title='Loss-Epoch', tools=tools,
                    x_axis_label='n of epoch', y_axis_label='loss', active_drag='pan', active_scroll="wheel_zoom")
    for i, color in zip(range(n), Spectral4):
        p1.line('x', 'loss' + str(i), line_width=2, source=source, color=color, alpha=0.8,
                legend=compare_name[i], line_dash="dashed")
    p1.line('x', 'loss', line_width=2, source=source, color='coral', legend=name)
    p1.title.text_font_size = "25px"
    p1.legend.click_policy = "hide"
    p1.legend.location = "top_right"
    p2 = plt.figure(plot_width=500, plot_height=400, y_axis_type="log", title='Error-Epoch', tools=tools,
                    x_range=p1.x_range, x_axis_label='n of epoch', y_axis_label='error', active_drag='pan',
                    active_scroll="wheel_zoom")
    for i, color in zip(range(n), Spectral4):
        p2.line('x', 'error' + str(i), line_width=2, source=source, color=color, alpha=0.8,
                legend=compare_name[i], line_dash="dashed")
    p2.line('x', 'error', line_width=2, source=source, color='coral', legend=name)
    p2.title.text_font_size = "25px"
    p2.legend.click_policy = "hide"
    p2.legend.location = "top_right"
    p = lyt.gridplot([[p1, p2]], toolbar_location="below", merge_tools=True)
    if valid is not None:
        tab1 = models.widgets.Panel(child=p, title="Test")
        loss, error = valid[0], valid[1]
        length = len(loss)
        x = list(range(length))
        data = dict(x=x, loss=loss, error=error)
        source = models.ColumnDataSource(data=data)
        v1 = plt.figure(plot_width=500, plot_height=400, y_axis_type="log", title='Loss-Epoch', tools=tools,
                        x_axis_label='n of epoch', y_axis_label='loss', active_drag='pan', active_scroll="wheel_zoom")
        v1.line('x', 'loss', line_width=2, source=source, color='coral', legend=name)
        v1.title.text_font_size = "25px"
        v1.legend.click_policy = "hide"
        v1.legend.location = "top_right"
        v2 = plt.figure(plot_width=500, plot_height=400, y_axis_type="log", title='Error-Epoch', tools=tools,
                        x_range=v1.x_range, x_axis_label='n of epoch', y_axis_label='error', active_drag='pan',
                        active_scroll="wheel_zoom")
        v2.line('x', 'error', line_width=2, source=source, color='coral', legend=name)
        v2.title.text_font_size = "25px"
        v2.legend.click_policy = "hide"
        v2.legend.location = "top_right"
        v = lyt.gridplot([[v1, v2]], toolbar_location="below", merge_tools=True)
        tab2 = models.widgets.Panel(child=v, title="Valid")
        tabs = models.widgets.Tabs(tabs=[tab1, tab2], sizing_mode="stretch_both")
        plt.save(tabs)
    else:
        plt.save(p)


def _compare_savefiles(x, y):
    xt = os.stat(x)
    yt = os.stat(y)
    if xt.st_mtime > yt.st_mtime:
        return 1
    else:
        return -1


def _load_data(file, part):
    return np.load(file)[part].tolist()


def _combine_repo(repo, compare_repo, part):
    compare_saving = []
    compare_name = []

    if compare_repo is None or compare_repo != []:
        compare_saving = [repo]
    if isinstance(compare_repo, (tuple, list)):
        compare_repo = [repo] + compare_repo
        for r in compare_repo:
            assert isinstance(r, str)
            savefile = './{}/save/final/final.npz'.format(r)
            if os.path.isfile(savefile):
                compare_saving.append(_load_data(savefile, part))
                compare_name.append(r)
            else:
                path = './{}/save'.format(r)
                savelist = [path + name for name in os.listdir(path) if name.endswith('.npz')]
                assert savelist > 0
                savelist.sort(_compare_savefiles)
                compare_saving.append(_load_data(savelist[-1], part))
    else:
        savelist = [compare_repo + name for name in os.listdir(compare_repo) if name.endswith('.npz')]
        for filename in savelist:
            compare_saving.append(_load_data(compare_repo + '/' + filename, part))
            compare_name.append(filename)
    return compare_saving[0], compare_saving[1:], compare_name[1:]


def progress(repo, compare_repo=None):
    repo_saving, compare_saving, compare_name = _combine_repo(repo, compare_repo, 'mainloop')
    monitor_progress(repo, 'progress.html', repo_saving['losses'], repo_saving['errors'],
                     compare_saving, compare_name, None)


def weight(repo, compare_repo=None):
    repo_saving, compare_saving, compare_name = _combine_repo(repo, compare_repo, 'trainable_params')
    def show(attr, old, new):
        layername=new
        g=[[]]
        for i, (wtname, wt) in enumerate(repo_saving[layername].items()):
            if wt.ndim == 4:
                wt.transpose(0, 2, 1, 3)
                wt.reshape([wt.shape[:2], wt.shape[2:]])
            p = plt.figure(plot_width=100, plot_height=100, title=wtname, tools=[])
            if wt.ndim == 1:
                p.line(range(wt.shape[0]), wt, line_width=2, color='black')
            else:
                p.image([wt], [0], [0], [p.x_range[-1]], [p.y_range[-1]])
            g[-1].append(p)
            if (i + 1) % 5 == 0:
                g.append([])
        v = lyt.gridplot(g, toolbar_location="below", merge_tools=True)

    plt.output_file('./weightplot.html', title='WeightPlot')
    select = models.widgets.Select(title="Layer:", value=repo_saving.keys()[0], options=repo_saving.keys())
    select.on_change("value",show)
    plt.save(lyt.widgetbox(select))
