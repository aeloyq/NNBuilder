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


def monitor_progress(current_saving, campare_saving=(), path='./'):
    def get_progress(saving_list):
        train_x_list, train_losses_list, valid_x_list, valid_losses_list, valid_scores_list, \
        test_x_list, test_losses_list, test_scores_list, names = [[], [], [], [], [], [], [], [], []]
        for saving in saving_list:
            names.append(saving['name'])
            train_losses_list.append(saving['train_losses'])
            train_x_list.append(range(1, len(saving['train_losses']) + 1))
            if 'EarlyStop' in saving['extensions']:
                valid_losses_list.append(saving['extensions']['EarlyStop']['valid_losses'])
                valid_scores_list.append(saving['extensions']['EarlyStop']['valid_scores'])
                valid_x_list.append(range(1,len(valid_losses_list[-1])+1))
            else:
                valid_losses_list.append(None)
                valid_scores_list.append(None)
                valid_x_list.append(None)
            test_losses_list.append(saving['losses'])
            test_scores_list.append(saving['scores'])
            test_x_list.append(range(len(test_losses_list[-1])))
        return train_x_list, train_losses_list, valid_x_list, valid_losses_list, valid_scores_list, test_x_list, test_losses_list, test_scores_list, names

    def plot_train(title, n_compare, train_x_list, train_losses_list):
        train_length = len(train_x_list[0])
        data = dict(x=train_x_list[0], loss=train_losses_list[0])
        for i in range(1, n_compare + 1):
            campare_length = len(train_losses_list[i])
            pad_length = max(train_length - campare_length, 0)
            data['loss' + str(i)] = train_losses_list[i] + [np.nan] * pad_length
        source = models.ColumnDataSource(data=data)
        train_p = plt.figure(plot_width=1000, plot_height=400, y_axis_type="log", title='Loss-Epoch', tools=tools,
                             x_axis_label='n of epoch', y_axis_label='loss', active_drag='pan',
                             active_scroll="wheel_zoom")
        for i, color in zip(range(n_compare), Spectral4):
            train_p.line('x', 'loss' + str(i), line_width=2, source=source, color=color, alpha=0.8,
                         legend=names[i], line_dash="dashed")
        train_p.line('x', 'loss', line_width=2, source=source, color='coral', legend=names[0])
        train_p.title.text_font_size = "25px"
        train_p.legend.click_policy = "hide"
        train_p.legend.location = "top_right"
        return models.widgets.Panel(child=train_p, title=title)

    def plot_valid(title, n_compare, valid_x_list, valid_losses_list, valid_scores_list, x_label):
        valid_length = len(valid_x_list[0])
        data = dict(x=valid_x_list[0], loss=valid_losses_list[0])
        for m in valid_scores_list[0]:
            data[m] = valid_scores_list[0][m]
        for i in range(n_compare):
            if valid_x_list[i] is not None:
                compare_length = len(valid_x_list[i])
                pad_length = max(valid_length - compare_length, 0)
                data['loss' + str(i)] = valid_losses_list[i] + [np.nan] * pad_length
                for m in valid_scores_list[0]:
                    if m in valid_scores_list[i]:
                        data[m + str(i)] = valid_scores_list[i][m] + [np.nan] * pad_length
        source = models.ColumnDataSource(data=data)
        valid_p1 = plt.figure(plot_width=1000, plot_height=400, y_axis_type="log", title='Loss-Epoch', tools=tools,
                              x_axis_label='n of epoch', y_axis_label='loss', active_drag='pan',
                              active_scroll="wheel_zoom")
        for i, color in zip(range(n_compare), Spectral4):
            if valid_x_list[i] is not None:
                valid_p1.line('x', 'loss' + str(i), line_width=2, source=source, color=color, alpha=0.8,
                              legend=names[i], line_dash="dashed")
        valid_p1.line('x', 'loss', line_width=2, source=source, color='coral', legend=names[0])
        valid_p1.title.text_font_size = "25px"
        valid_p1.legend.click_policy = "hide"
        valid_p1.legend.location = "top_right"
        gridplot = [[valid_p1]]
        for m in valid_scores_list[0]:
            valid_p2 = plt.figure(plot_width=1000, plot_height=400, title='{}-Epoch'.format(m), tools=tools,
                                  x_range=valid_p1.x_range, x_axis_label='n of %s' % x_label, y_axis_label=m,
                                  active_drag='pan',
                                  active_scroll="wheel_zoom")
            for i, color in zip(range(n_compare), Spectral4):
                if m in valid_scores_list[i]:
                    valid_p2.line('x', m + str(i), line_width=2, source=source, color=color, alpha=0.8,
                                  legend=names[i], line_dash="dashed")
            valid_p2.line('x', m, line_width=2, source=source, color='coral', legend=names[0])
            valid_p2.title.text_font_size = "25px"
            valid_p2.legend.click_policy = "hide"
            valid_p2.legend.location = "top_right"
            gridplot.append([valid_p2])
        valid_p = lyt.gridplot(gridplot, toolbar_location="below", merge_tools=True)
        return models.widgets.Panel(child=valid_p, title=title)

    plt.output_file(path, title='Training Progress')
    tools = "pan,wheel_zoom,reset,save"

    train_x_list, train_losses_list, valid_x_list, valid_losses_list, valid_scores_list, test_x_list, test_losses_list, test_scores_list, names = get_progress(
        [current_saving] + campare_saving)
    n_compare = len(campare_saving)
    tabs = []
    # train #
    tabs.append(plot_train('Train', n_compare, train_x_list, train_losses_list))
    # valid #
    if valid_x_list[0] is not None:
        tabs.append(plot_valid('Valid', n_compare, valid_x_list, valid_losses_list, valid_scores_list, 'valid'))
    # test #
    tabs.append(plot_valid('Test', n_compare, test_x_list, test_losses_list, test_scores_list, 'epoch'))

    page = models.widgets.Tabs(tabs=tabs, sizing_mode="stretch_both")
    plt.save(page)


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

    if compare_repo is None or compare_repo != []:
        compare_saving = [repo]
    if isinstance(compare_repo, (tuple, list)):
        compare_repo = [repo] + compare_repo
        for r in compare_repo:
            assert isinstance(r, str)
            savefile = './{}/save/final/final.npz'.format(r)
            if os.path.isfile(savefile):
                compare_saving.append(_load_data(savefile, part))
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
    return compare_saving


def progress(repo, compare_repo=None):
    saves = _combine_repo(repo, compare_repo, 'mainloop')
    monitor_progress(saves[0], saves[1:], 'progress.html')


def weight(repo, compare_repo=None):
    repo_saving, compare_saving, compare_name = _combine_repo(repo, compare_repo, 'trainable_params')

    def show(attr, old, new):
        layername = new
        g = [[]]
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
    select.on_change("value", show)
    plt.save(lyt.widgetbox(select))
