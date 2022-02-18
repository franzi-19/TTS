import os

import umap
from bokeh.io import save, export_png
from bokeh.models import (BoxZoomTool, ColorBar, ColumnDataSource, HoverTool,
                          OpenURL, ResetTool, TapTool)
from bokeh.palettes import Category10, Category20, Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, linear_cmap

CIRCLE_SIZE = 2
CROSS_SIZE = 2
TICK_NUMBER_SIZE = "15pt"

def plot_embeddings(embeds, plot_path, labels=None, filename='plot.png'): # embeds: array([array([coordinate1, co2]), ...])
    model = umap.UMAP()
    projection = model.fit_transform(embeds)

    if labels != None and labels != []:
        source_wav_stems = ColumnDataSource(
                data=dict(
                    x = projection.T[0].tolist(),
                    y = projection.T[1].tolist(),
                    #desc=locations,
                    label=labels
                )
            )
    else:
        source_wav_stems = ColumnDataSource(
                data=dict(
                    x = projection.T[0].tolist(),
                    y = projection.T[1].tolist(),
                    #desc=locations
                )
            )

    # hover = HoverTool(
    #         tooltips=[
    #             #("file", "@desc"),
    #             ("attack", "@label"),
    #         ]
    #     )

    if labels != None and labels != []:
        factors = list(set(labels))
        pal_size = max(len(factors), 3)
        if pal_size <= 10:
            pal = Category10[pal_size]
        elif pal_size <= 20:
            pal = Category20[pal_size]
        else:
            raise ValueError('Too many different labels to plot')

    p = figure(plot_width=1000, plot_height=600)#, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])

    if labels != None and labels != []:
        p.circle('x', 'y',  source=source_wav_stems, size=CIRCLE_SIZE, color=factor_cmap('label', palette=pal, factors=factors))#, legend_group='label')
        # p.legend.location = "bottom_left"
        # p.legend.click_policy= "hide"
    else:
        p.circle('x', 'y',  source=source_wav_stems)

    p.xaxis.major_label_text_font_size = TICK_NUMBER_SIZE
    p.yaxis.major_label_text_font_size = TICK_NUMBER_SIZE

    file_path = plot_path + filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # save(p, file_path, title=title)
    export_png(p, filename=file_path)
    print(f'saved plot at {file_path}')

# https://docs.bokeh.org/en/latest/docs/user_guide/styling.html#using-mappers
def plot_embeddings_continuous(embeds, plot_path, labels, title=None):
    model = umap.UMAP()
    projection = model.fit_transform(embeds)

    if isinstance(labels[0], str):
        factors = list(set(labels))
        pal_size = max(len(factors), 3)
        pal = Category10[pal_size]
        mapper = factor_cmap('label', palette=pal, factors=factors)
    else:
        mapper = linear_cmap(field_name='label', palette=Spectral6 ,low=min(labels) ,high=max(labels))
    source = ColumnDataSource(dict(x=projection.T[0].tolist(), y=projection.T[1].tolist(), label=labels))
    hover = HoverTool(tooltips=[("attack", "@label")])

    p = figure(plot_width=1000, plot_height=600, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])
    p.circle(x='x', y='y', line_color=mapper, color=mapper, source=source)
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8)
    p.add_layout(color_bar, 'right')

    if title: filename = f"plot_{title}.html"
    else: filename = f"{filename}.html"

    file_path = plot_path + filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save(p, file_path, title=title)
    print(f'saved plot at {file_path}')


def plot_two_sets(embeds_1, labels_1, embeds_2, labels_2, plot_path, filename='plot.png'):
    model = umap.UMAP()
    len_1 = len(embeds_1)
    projection = model.fit_transform(embeds_1 + embeds_2)

    data_1 = ColumnDataSource(data=dict(x = projection.T[0].tolist()[:len_1], y = projection.T[1].tolist()[:len_1], label=labels_1))    # , desc=locations_1, label=labels_1))
    data_2 = ColumnDataSource(data=dict(x = projection.T[0].tolist()[len_1:], y = projection.T[1].tolist()[len_1:], label=labels_2))    # , desc=locations_2, label=labels_2))

    # hover = HoverTool(tooltips=[ ("attack", "@label"),])   # [("file", "@desc"), ("speaker", "@label"),])

    p = figure(plot_width=1000, plot_height=600) #, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])

    factors_1 = list(set(labels_1))
    pal_size_1 = max(len(factors_1), 3)
    pal_1 = Category20[pal_size_1]

    p.circle('x', 'y',  source=data_1, color=factor_cmap('label', palette=pal_1, factors=factors_1), size=CIRCLE_SIZE)#, legend_group='label')
    p.cross('x', 'y',  source=data_2, color=factor_cmap('label', palette=pal_1, factors=factors_1), size=CROSS_SIZE)#, legend_group='label')

    # p.legend.location = "top_left"
    # p.legend.click_policy= "hide"
    # p.legend.label_text_font_size = "5pt"

    p.xaxis.major_label_text_font_size = TICK_NUMBER_SIZE
    p.yaxis.major_label_text_font_size = TICK_NUMBER_SIZE

    file_path = plot_path + filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # save(p, file_path, title=title)
    export_png(p, filename=file_path)
    print(f'saved plot at {file_path}')

def just_plot(x, y, plot_path, labels, title=None):
    hover = HoverTool(tooltips=[ ("attack", "@label"),])
    source_wav_stems = ColumnDataSource(data=dict(x=x, y=y, label=labels))

    factors = list(set(labels))
    pal_size = max(len(factors), 3)
    pal = Category20[pal_size]

    p = figure(plot_width=1000, plot_height=600, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])
    
    p.circle('x', 'y',  source=source_wav_stems, color=factor_cmap('label', palette=pal, factors=factors), legend_group='label')
    p.legend.location = "bottom_right"
    p.legend.click_policy= "hide"

    if title: filename = f"plot_{title}.html"
    else: filename = "plot.html"

    file_path = plot_path + filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save(p, file_path, title=title)
    print(f'saved plot at {file_path}')

def plot_centroids(points, point_labels, centroids, centroid_labels, centroid_colors, plot_path, title):
    hover = HoverTool(tooltips=[ ("attack", "@label"),])

    data_points = ColumnDataSource(data=dict(x=[point[0] for point in points], y=[point[1] for point in points], label=point_labels))
    data_centroids = ColumnDataSource(data=dict(x=[cent[0] for cent in centroids], y=[cent[1] for cent in centroids], label=[f"{centroid_colors[idx]} ({label[0]},{label[1]})" for idx, label in enumerate(centroid_labels)], color=centroid_colors))

    factors = list(set(point_labels))
    pal_size = max(len(factors), 3)
    pal = Category20[pal_size]

    p = figure(plot_width=1000, plot_height=600, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])
    
    p.circle('x', 'y',  source=data_points, color=factor_cmap('label', palette=pal, factors=factors), legend_group='label')
    p.cross('x', 'y',  source=data_centroids, color=factor_cmap('color', palette=pal, factors=factors), size=15)#,legend_group='color')
    p.legend.location = "bottom_right"
    p.legend.click_policy= "hide"

    if title: filename = f"plot_{title}.html"
    else: filename = "plot.html"

    file_path = plot_path + filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save(p, file_path, title=title)
    print(f'saved plot at {file_path}')
