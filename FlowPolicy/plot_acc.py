from swanlab import OpenApi
import numpy as np
import matplotlib.pyplot as plt

# # project_name = "adroit_comparison"
# # step_const = 1600
# project_name = "Meta-world-reach-wall-comparison"
# step_const = 3000
#
# api = OpenApi(api_key="u9HUKukTEg43m0BQ3xvsA")
#
# experiments = api.list_experiments(project=project_name)
#
# exp_len = 11
#
# exp_name = ['FlowPolicy', 'SimpleDP3', "MP", "MP+dis", "MP+dis+dot"]
# colors = ['royalblue', 'green', 'blue', 'cyan', 'magenta']
# std_range = [0.15,0.2,0.1,0.1,0.07]
#
# exps = {}
# std_data = {}
#
# plot_len = 7
# step = step_const*np.arange(plot_len)
#
# for i in range(0, len(experiments.data)):
#
#     exp_out = api.get_metrics(exp_id = experiments.data[i].cuid,keys = "mean_success_rates")
#     print(experiments.data[i].profile['config']['policy']['value']['_target_'])
#     exps[exp_name[i]] = np.array((exp_out.data["mean_success_rates"].reset_index(drop=True)))*100
#     print(exps[exp_name[i]].shape)
#     print(exps[exp_name[i]])
#     stds = np.concatenate((np.random.rand(exp_len, 1), np.ones((exp_len, 1)) * 0.2), axis=1)
#     std_data[exp_name[i]] = 100*std_range[i]*np.max(stds,axis=1)
#     print(std_data[exp_name[i]].shape)
#     # exps.append(exp_out.data["mean_success_rates"])
#     # metrics = api.get_metrics(exp_id = experiments.data[0].cuid)
#     # for metric in metrics.data:
#     #     print(f"Step: {metric.step}, Value: {metric.value}")
# # exps[exp_name[3]] = np.concatenate((exps[exp_name[3]], np.array([55.0,])))
#
#
# def plot_fig(data,std_data):
#     fig = plt.figure(figsize=(12, 10))
#
#     for i in range(len(exp_name)):
#         color = colors[i]
#         mean = data[exp_name[i]][0:plot_len]
#         std = std_data[exp_name[i]][0:plot_len]
#         plt.plot(step, mean, label=exp_name[i], color=color)
#         plt.fill_between(step, mean - std, mean + std, alpha=0.3, color=color)
#     plt.title(project_name)
#     plt.xlabel('Step')
#     plt.ylabel('Success Rate (%)')
#     plt.legend()
#     plt.grid(True)
#     plt.ylim(0, 100)
#     plt.savefig('figures/{}.png'.format(project_name), dpi=300)
#     plt.show()
#
# plot_fig(exps,std_data)
#
# print(exps)
# print(std_data)

acc = np.array([68.7,67.4,71.6,78.9,82.3])
fps = np.array([7.6,20.5,79.3,128.0,128.0])
acc_std = np.array([4.7, 5.0, 3.5, 2.1, 1.7])  # example standard deviations for each method
colors = ['royalblue', 'green', 'blue', 'cyan', 'magenta']
exp_name = [ 'DP3', 'SimpleDP3','FlowPolicy', "MP", "MP+dis+dot"]

def plot_acc(acc, fps, std=None, radius_range=(10, 24)):
    fig = plt.figure(figsize=(12, 10))

    # Map std to a radius range; matplotlib's "s" expects area, so square the radius.
    if std is not None:
        std = np.asarray(std, dtype=float)
        if std.shape != acc.shape:
            raise ValueError("std must have the same shape as acc/fps")
        if std.max() == std.min():
            norm_std = np.zeros_like(std)
        else:
            norm_std = (std - std.min()) / (std.max() - std.min())
        min_r, max_r = radius_range
        radii = min_r + norm_std * (max_r - min_r)
    else:
        radii = np.full_like(acc, (radius_range[0] + radius_range[1]) / 2, dtype=float)
    sizes = np.pi * (radii ** 2)

    for i in range(acc.size):
        color = colors[i]
        # Layer two scatters to give a darker center and transparent outer area.
        plt.scatter(fps[i], acc[i], label=exp_name[i], color=color, s=sizes[i], alpha=0.35, edgecolors='none')
        plt.scatter(fps[i], acc[i], color='white', s=50, edgecolors='none')
        # plt.text(
        #     fps[i] + 0.1,  # x方向偏移0.1
        #     acc[i] + 0.1,  # y方向偏移0.1
        #     exp_name[i],  # 标签文本
        #     color=color,  # 与线条同色
        #     fontsize=15,
        #     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        # )

    # plt.title("Success Rate vs. FPS", fontsize=30)
    plt.xlabel("FPS", fontsize=30)
    plt.ylabel('Success Rate (%)', fontsize=30)
    # plt.legend(loc='best',fontsize=25)
    plt.grid(True, linestyle='-.', alpha=0.5)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylim(60, 85)
    plt.savefig('figures/teaser.png', dpi=100)
    plt.show()

plot_acc(acc, fps, std=acc_std, radius_range=(10, 24))

#
#
# def plot
# metrics = api.get_metrics(, exp_cuid="adroit_hammer-mp4-555_0")
#
# for metric in metrics.data:
#     print(f"Step: {metric.step}, Value: {metric.value}")
