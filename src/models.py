import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import grad as torch_grad

from helpers import get_dimwise_prob_metrics, save_current_plot, score_real_fake, make_dimwise_probability_plot, \
    make_dimwise_prediction_performance_plot, make_cat_dist_plots, make_num_dist_plots, score_oversampling_performance, \
    generate_date_prefix, get_cat_dims


class BaseGAN():
    def __init__(self,
                 netG=None,
                 netD=None,
                 g_optim=None,
                 d_optim=None,
                 d_updates_per_g: int = 3,
                 verbose: int = 1,
                 write_to_disk: bool = True,
                 print_every: int = 150,
                 compute_metrics_every: int = 150,
                 plot_every: int = 300,
                 save_model_every: int = 5000,
                 save_data_every: int = 10000,
                 prefix: str = None,
                 transformer=None,
                 num_cols=None,
                 cat_cols=None,
                 cat_dims=None):

        self.netG = netG
        self.netD = netD
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.d_updates_per_g = d_updates_per_g
        self.verbose = verbose
        self.print_every = print_every
        self.compute_metrics_every = compute_metrics_every
        self.plot_every = plot_every
        self.save_model_every = save_model_every
        self.save_data_every = save_data_every
        self.write_to_disk = write_to_disk
        self.prefix = prefix

        if self.prefix is None:
            prefix = generate_date_prefix()
            self.prefix = f'Experiments/results/{prefix}'

        if self.write_to_disk:

            Path(self.prefix).mkdir(parents=True, exist_ok=True)
            for subfolder in ['plots', 'dataset', 'models', 'models/netG', 'models/netD', 'metrics']:
                Path(self.prefix + f'/{subfolder}').mkdir(parents=True, exist_ok=True)

        self.transformer = transformer
        self.num_cols = num_cols
        self.num_dim = len(num_cols) if num_cols is not None else None
        self.cat_cols = cat_cols
        self.cat_dims = cat_dims
        self.total_iters = 0
        self.total_gen_iters = 0
        self.metrics_to_use = self._get_list_of_metrics()
        self.metrics = {metric: list() for metric in self.metrics_to_use}

    def _init_netG(self, kwargs=dict()):

        netG = Generator(cat_output_dims=self.cat_dims, output_dim=self.num_dim, **kwargs)
        return netG

    def _init_netD(self, kwargs=dict()):
        netD = Discriminator(cat_input_dims=self.cat_dims, input_dim=self.num_dim, **kwargs)
        return netD

    def _init_optim_default(self, net, lr=1e-4):
        return optim.Adam(net.parameters(), lr=lr, betas=(.0, .9))

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            num_cols: list = None,
            cat_cols: list = None,
            cat_dims: list = None,
            condition: bool = True,
            netG_kwargs: dict = dict(),
            netD_kwargs: dict = dict(),
            batch_size: int = 256,
            n_iters: int = None,
            epochs: int = None):

        if cat_cols is not None and type(X).__name__ == 'DataFrame':
            cat_dims = get_cat_dims(X, cat_cols)

        if self.num_dim is None:
            self.num_dim = len(num_cols) if num_cols is not None else X.shape[1]
        if cat_dims is not None:
            self.cat_dims = cat_dims

        X_tens = torch.Tensor(X) if type(X).__name__ != 'DataFrame' else torch.Tensor(X.values)
        y_tens = torch.Tensor(y).view(-1, 1) if y is not None else None

        if self.netG is not None:
            self.condition = self.netG.condition
        elif self.netD is not None:
            self.condition = self.netD.condition
        else:
            self.condition = condition

        y_train = None
        if not self.condition:
            y_tens = torch.zeros(y_tens.size()[0], 2).scatter_(1, y_tens.long(), 1)
            X_tens = torch.cat([X_tens, y_tens], dim=1)
            y_tens = None
            self.cat_dims = self.cat_dims + [2] if self.cat_dims is not None else [2]
            y_train = y

        if self.netG is None:
            netG_kwargs['condition'] = self.condition
            self.netG = self._init_netG(kwargs=netG_kwargs)
            self.g_optim = self._init_optim_default(net=self.netG)
        if self.netD is None:
            netD_kwargs['condition'] = self.condition
            self.netD = self._init_netD(kwargs=netD_kwargs)
            self.d_optim = self._init_optim_default(net=self.netD)

        #
        if epochs is None and n_iters is None:
            n_iters = 1000
        elif epochs is not None:
            iters_per_epoch = int(np.ceil(X_tens.size()[0] / batch_size))
            n_iters = int(iters_per_epoch * epochs)

        self.train(X=X_tens, y=y_tens, batch_size=batch_size, n_iters=n_iters, y_train=y_train)

        return self

    def train(self, X, y=None, batch_size=256, n_iters=1000, y_train=None):

        self.batch_size = batch_size
        self.target_batch_size = batch_size
        self.n_iters = n_iters
        iters_per_epoch = int(np.ceil(X.size()[0] / self.batch_size))

        if n_iters < iters_per_epoch:
            logging.warning(
                f'n_iters={n_iters} but it would take at least {iters_per_epoch} total_iters to complete one epoch.')

        epochs_needed = int(np.ceil(n_iters / iters_per_epoch))

        for epoch in range(epochs_needed):
            self._train_epoch(X=X, y=y, iters_per_epoch=iters_per_epoch)

        #### END of training
        if self.print_every > 0:
            self._print_metrics(n_iters=self.n_iters, end='\n')

        if self.write_to_disk:
            self._plot_metrics()
            save_current_plot(path=self.prefix, name=f'metrics_final_iters_{self.total_iters}', show=True)
            self._save_data(self.sample(n=50000, y='50-50'))
            self._save_metrics()
            self._save_models()
            self.netG._remove_activation_functions()
            joblib.dump(self.__dict__, f'{self.prefix}/models/_whole_basegan.pkl')
            self.netG._restore_activation_functions()

        return self

    def _train_epoch(self, X, y, iters_per_epoch):

        permutation = torch.randperm(X.size()[0])
        X = X[permutation]
        if y is not None:
            y = y[permutation]
        self.batch_size = self.target_batch_size

        for batch_idx in range(iters_per_epoch):

            X_batch = X[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
            if y is not None:
                y_batch = y[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
            else:
                y_batch = None
            self.batch_size = X_batch.size()[0]

            self.netG.train()
            self.netD.train()

            self._netD_iter(X_batch, y_batch)

            if batch_idx % self.d_updates_per_g == 0:
                self._netG_iter(X_batch, y_batch)

            self.netG.eval()
            self.netD.eval()

            if self.total_iters % self.compute_metrics_every == 0 and self.compute_metrics_every > 0:
                self._compute_metrics_callback(X=X, y=y)

            if all([self.total_iters % self.print_every == 0, self.total_iters > 0, self.print_every > 0]):
                self._print_metrics(n_iters=self.n_iters, end='\n')
            elif self.total_iters % 500 == 0 and self.print_every > 0:
                self._print_metrics(n_iters=self.n_iters, end='\r')

            if self.write_to_disk:
                if self.total_iters % self.plot_every == 0 and self.total_iters > 0:
                    self._plotting_callback(X=X)

                if self.total_iters % self.save_model_every == 0 and self.total_iters > 0:
                    self._save_models()
                if self.total_iters % self.save_data_every == 0 and self.total_iters > 0:
                    self._save_data(self.sample(n=25000, y='50-50'))

            self.total_iters += 1

    def _netD_iter(self, X_batch, y_batch=None):

        self.netD.zero_grad()

        output = self.netD(X_batch, y_batch).view(-1)

        lossD_real = F.binary_cross_entropy(output, torch.ones(self.batch_size))
        lossD_real.backward()
        D_x = output.mean().item()

        fake = self.netG.sample(self.batch_size, y=y_batch)

        output = self.netD(fake, y_batch).view(-1)
        lossD_fake = F.binary_cross_entropy(output, torch.zeros(self.batch_size))
        lossD_fake.backward()
        D_G_z = output.mean().item()

        lossD = lossD_fake + lossD_real
        self.d_optim.step()

        self.metrics['total_iters'].append(self.total_iters)
        self.metrics['netD_loss'].append(lossD.item())
        self.metrics['avg_D_real'].append(D_x)
        self.metrics['avg_D_fake'].append(D_G_z)

    def _netG_iter(self, X_batch, y_batch=None):

        self.netG.zero_grad()

        fake = self.netG.sample(self.batch_size, y=y_batch)
        output = self.netD(fake, y_batch).view(-1)
        lossG = F.binary_cross_entropy(output, torch.ones(self.batch_size))
        lossG.backward()

        self.g_optim.step()

        self.total_gen_iters += 1
        for _ in range(self.d_updates_per_g):
            self.metrics['total_gen_iters'].append(self.total_gen_iters)
            self.metrics['netG_loss'].append(lossG.item())

    def sample(self, n=5000, y=None, as_numpy=True):

        q, remainder = divmod(n, 1000)

        if remainder != 0:
            if y is not None and not isinstance(y, (int, float, str)):
                X = self.netG.sample(n=remainder, y=y[:remainder]).detach().numpy()
            else:
                X = self.netG.sample(n=remainder, y=y).detach().numpy()

        for i in range(q):
            if y is not None and not isinstance(y, (int, float, str)):
                X_curr = self.netG.sample(n=1000, y=y[remainder + i * 1000:remainder + (i + 1) * 1000]).detach().numpy()
            else:
                X_curr = self.netG.sample(n=1000, y=y).detach().numpy()
            try:
                X = np.vstack([X, X_curr])
            except UnboundLocalError:
                X = X_curr
        return X

    def resample(self, X, y):

        n_minority_needed = int(((1 - y).sum() - y.sum()))
        print("needed sample", n_minority_needed)

        if self.netG.condition:

            X_fake = self.sample(n_minority_needed, y=1)
            y_fake = np.ones(n_minority_needed)
            X_os = np.vstack([X, X_fake])
            y_os = np.hstack([y, y_fake])
        else:

            synth_imb_ratio = self.sample(n=int(1200))[:, -1].mean()
            if synth_imb_ratio < 0.0001:
                raise ValueError(f'Too few minority cases are generated. Only {synth_imb_ratio * 100:.2f}% '
                                 f'are minority cases.')

            n_synth_needed = int(1.2 * ((1 - y).sum() - y.sum()) / synth_imb_ratio)
            X_y_fake = self.sample(n=int(n_synth_needed))

            X_fake, y_fake = np.hsplit(X_y_fake, [-2])
            y_fake = y_fake[:, 1]
            X_fake = self._clean_output(X_fake[y_fake.flatten() == 1])
            y_fake = y_fake[y_fake.flatten() == 1]
            X_os = np.vstack([X, X_fake[:n_minority_needed]])
            y_os = np.hstack([y, y_fake[:n_minority_needed]])

        return X_os, y_os

    def clean_sample(self, n=5000, y=None):
        X = self.sample(n=n, as_numpy=True, y=y)
        X = self._clean_output(X)
        return X

    def _clean_output(self, X):

        X = X.clip(0, 1)

        return X

    def _compute_metrics_callback(self, X, y=None):

        if y is None:
            X_fake = self.sample(n=int(X.size()[0] * 2), as_numpy=False)
        else:
            n = int(X.size()[0] * 1.5)
            y_fake = torch.cat([torch.zeros(n - (n // 2), 1), torch.ones(n // 2, 1)])
            X_fake = self.sample(n=n, y=y_fake, as_numpy=False)

        for measure in ['avg', 'std']:
            dimwise_prob_metrics = get_dimwise_prob_metrics(X_real=X, X_fake=X_fake, measure=measure,
                                                            n_num_cols=self.netG.output_dim)
            for value, name in zip(dimwise_prob_metrics,
                                   ['rmse', 'corr', 'rmse_num', 'corr_num', 'rmse_cat', 'corr_cat']):
                self.metrics[f'{name}_{measure}'].append(value)

        rf_scores = score_real_fake(X_real=X[:2000, :], X_fake=X_fake[:2000, :].clip(0, 1),
                                    classifier='rfc_shallow')
        for metric in ['auc', 'acc', 'f1']:
            self.metrics[f'real_fake_{metric}'].append(rf_scores[metric])

        if self.total_iters % (self.compute_metrics_every * 4) == 0:
            if y is None:

                comb_scores, fakeonly_scores = score_oversampling_performance(X_y_real=X, X_y_fake=X_fake.clip(0, 1))
                for metric in ['auc', 'acc', 'f1']:
                    self.metrics[f'oversampling_{metric}'].append(comb_scores[metric])
                    self.metrics[f'synthtraining_{metric}'].append(fakeonly_scores[metric])
            else:
                comb_scores, fakeonly_scores = score_oversampling_performance(X_y_real=X, X_y_fake=X_fake.clip(0, 1),
                                                                              y_real=y, y_fake=y_fake)
                for metric in ['auc', 'acc', 'f1']:
                    self.metrics[f'oversampling_{metric}'].append(comb_scores[metric])
                    self.metrics[f'synthtraining_{metric}'].append(fakeonly_scores[metric])

    def _plotting_callback(self, X, use_full_data: bool = False, y=None):
        if use_full_data:
            n = X.size()[0]
        else:
            n = 5000

        X_fake = self.sample(n=n, y='50-50', as_numpy=False)

        self._plot_metrics()
        save_current_plot(path=self.prefix + '/plots', name=f'metrics_latest_iter',
                          show=True)

        make_num_dist_plots(X_real=X.numpy(), X_fake=X_fake, show=False, num_cols=self.num_cols)
        save_current_plot(path=self.prefix + '/plots', name=f'num_dist_plots__iter{self.total_iters}',
                          show=True)

        if self.transformer is not None:
            make_cat_dist_plots(X_real=X.numpy(), X_fake=X_fake,
                                ohe=self.transformer,
                                num_cols=self.num_cols, cat_cols=self.cat_cols,
                                show=False)
            save_current_plot(path=self.prefix + '/plots', name=f'cat_dist_plots__iter{self.total_iters}',
                              show=True)

        if self.total_iters % (self.plot_every * 2) == 0:
            rmse_value, corr_value = self._plot_scatterplots(X_real=X.numpy()[:n, :], X_fake=X_fake,
                                                             show=False)
            save_current_plot(path=self.prefix + '/plots', name=f'scatterplots__iter{self.total_iters}',
                              show=True)
            self.metrics['rmse_pred_scatter'].append(rmse_value)
            self.metrics['rmse_pred_scatter'].append(rmse_value)
            self.metrics['corr_pred_scatter'].append(corr_value)
            self.metrics['corr_pred_scatter'].append(corr_value)

        self._plot_classification_metrics(show=False)
        save_current_plot(path=self.prefix + '/plots', name=f'classification_plots_latest_iter',
                          show=True)

    def _save_models(self):

        self.netG._remove_activation_functions()
        try:
            torch.save(self.netG, f'{self.prefix}/models/netG/netG_iter{self.total_iters}.statedict')
        except:
            logging.warning('Pickling netG failed.')
        self.netG._restore_activation_functions()
        try:
            torch.save(self.netD, f'{self.prefix}/models/netD/netD_iter{self.total_iters}.statedict')
        except:
            logging.warning('Pickling netD failed.')

    def _save_data(self, data):

        joblib.dump(data, f'{self.prefix}/dataset/data_arr_iter{self.total_iters}.pkl')

    def _save_metrics(self):

        joblib.dump(self.metrics, f'{self.prefix}/metrics{self.total_iters}.pkl')

    def _print_metrics(self, n_iters, end='\n'):

        out = f"[{self.total_iters:5}/{n_iters}] LossG: {self.metrics['netG_loss'][-1]:.3f} " \
              f"LossD: {self.metrics['netD_loss'][-1]:.3f} "

        try:
            out += f"RMSE AVG: {self.metrics['rmse_avg'][-1]:.3f} " \
                   f"NUM: {self.metrics['rmse_num_avg'][-1]:.3f} " \
                   f"SynTraiAuc: {self.metrics['synthtraining_auc'][-1]:.3f} " \
                   f"RFAcc: {self.metrics['real_fake_acc'][-1]:.3f}  "
        except IndexError:
            pass

        print(out, end=end)

    def _plot_metrics(self, show=False):
        fig, axes = plt.subplots(2, 2)
        axes = axes.flatten()
        fig.set_size_inches((16, 6))
        legend_kwargs = {'fontsize': 'x-small', 'title_fontsize': 'small', 'loc': 'upper left'}

        axes[0].plot(self.metrics['netG_loss'], label='Gen')
        axes[0].plot(self.metrics['netD_loss'], label='Disc')
        axes[0].legend(title='Loss', **legend_kwargs)
        axes[0].set_xticks([])

        axes[2].plot(self.metrics['avg_D_real'], label='Real')
        axes[2].plot(self.metrics['avg_D_fake'], label='Fake')
        axes[2].legend(title='Mean(D)', **legend_kwargs)

        axes[1].plot(self.metrics['rmse_avg'], label='All columns')
        axes[1].plot(self.metrics['rmse_num_avg'], label='Num')
        axes[1].plot(self.metrics['rmse_cat_avg'], label='Cat')
        axes[1].plot(self.metrics['rmse_std'], label='All columns', linestyle=':')
        axes[1].plot(self.metrics['rmse_num_std'], label='Num', linestyle=':')
        axes[1].plot(self.metrics['rmse_cat_std'], label='Cat', linestyle=':')
        axes[1].plot(self.metrics['rmse_pred_scatter'], label='Pred')
        axes[1].legend(title='RMSE AVG/STD', **legend_kwargs)
        axes[1].set_xticks([])
        axes[1].set_ylim(top=0.3)

        axes[3].plot(self.metrics['corr_avg'], label='All columns')
        axes[3].plot(self.metrics['corr_num_avg'], label='Num')
        axes[3].plot(self.metrics['corr_cat_avg'], label='Cat')
        axes[3].plot(self.metrics['corr_std'], label='All columns', linestyle=':')
        axes[3].plot(self.metrics['corr_num_std'], label='Num', linestyle=':')
        axes[3].plot(self.metrics['corr_cat_std'], label='Cat', linestyle=':')
        axes[3].legend(title='CORR AVG/STD', **legend_kwargs)
        axes[3].set_ylim(bottom=0.8)

        plt.tight_layout()
        if show:
            plt.show()

    def _plot_scatterplots(self, X_real, X_fake, show=False):
        fig, axes = plt.subplots(1, 3)
        fig.set_size_inches((12, 4))
        make_dimwise_probability_plot(X_real=X_real, X_fake=X_fake, measure='mean',
                                      show=False, make_fig=False, ax=axes[0])
        axes[0].set_title('Mean')
        make_dimwise_probability_plot(X_real=X_real, X_fake=X_fake, measure='std',
                                      show=False, make_fig=False, ax=axes[1])
        axes[1].set_title('Standard Deviation')
        axes[1].set_ylabel(None)
        rmse_value, corr_value = make_dimwise_prediction_performance_plot(X_real=X_real, X_fake=X_fake,
                                                                          n_num_cols=self.num_dim,
                                                                          cat_input_dims=self.netD.cat_input_dims,
                                                                          show=False, make_fig=False, ax=axes[2])
        axes[2].set_title('Prediction Performance')
        axes[2].set_ylabel(None)

        plt.tight_layout()
        if show:
            plt.show()
        return rmse_value, corr_value

    def _plot_classification_metrics(self, show=False):
        fig, axes = plt.subplots(3, 1)
        fig.set_size_inches((8, 5))
        axes[0].plot(self.metrics['oversampling_acc'], label='acc')
        axes[0].plot(self.metrics['oversampling_auc'], label='auc')
        axes[0].plot(self.metrics['oversampling_f1'], label='f1')
        axes[0].legend(title='Oversampling training')

        axes[1].plot(self.metrics['synthtraining_acc'], label='acc')
        axes[1].plot(self.metrics['synthtraining_auc'], label='auc')
        axes[1].plot(self.metrics['synthtraining_f1'], label='f1')
        axes[1].legend(title='Synth training')

        axes[2].plot(self.metrics['real_fake_acc'], label='acc')
        axes[2].plot(self.metrics['real_fake_auc'], label='auc')
        axes[2].plot(self.metrics['real_fake_f1'], label='f1')
        axes[2].legend(title='Real/fake')

        plt.tight_layout()
        if show:
            plt.show()

    def get_metrics_df(self):
        return pd.DataFrame.from_dict(self.metrics, orient='index').T

    def get_report_row(self, return_df: bool = True):
        report_dict = {}
        for name in ['prefix', 'total_iters', 'total_gen_iters', 'd_updates_per_g', 'target_batch_size',
                     'optimG.betas', 'optimD.betas', 'optimG.lr', 'optimD.lr',
                     'netG.n_hidden_layers', 'netG.hidden_layer_sizes', 'netG.n_cross_layers',
                     'netD.n_hidden_layers', 'netD.hidden_layer_sizes', 'netD.n_cross_layers',
                     'netG.noise_dim', 'netG.normal_noise', 'netG.condition', 'netG.cat_activation', 'netG.dropout',
                     'netG.condition_num_on_cat', 'netG.num_activation', 'netG.cat_activation', 'netG.gumbel_kwargs',
                     'netG.activation_name', 'netG.layer_norm',
                     'netD.embedding_dims', 'netD.condition', 'netD.sigmoid_activation', 'netD.layer_norm',
                     'netD.noisy_num_cols', 'netD.activation_name',
                     'metrics.netG_loss', 'metrics.netD_loss',
                     'metrics.rmse_avg', 'metrics.corr_avg',
                     'metrics.rmse_std', 'metrics.corr_std',
                     'metrics.rmse_num_avg', 'metrics.rmse_num_std',
                     'metrics.corr_num_avg', 'metrics.corr_num_std',
                     'metrics.rmse_cat_avg', 'metrics.rmse_cat_std',
                     'metrics.corr_cat_avg', 'metrics.corr_cat_std',
                     'metrics.rmse_pred_scatter', 'metrics.corr_pred_scatter',
                     'metrics.real_fake_acc', 'metrics.real_fake_auc', 'metrics.real_fake_f1',
                     'metrics.oversampling_acc', 'metrics.oversampling_auc', 'metrics.oversampling_f1',
                     'metrics.synthtraining_acc', 'metrics.synthtraining_auc', 'metrics.synthtraining_f1']:
            try:
                if 'netG' in name[:5]:
                    report_dict[name] = self.netG.__getattribute__(name[5:])
                elif 'netD' in name[:5]:
                    report_dict[name] = self.netD.__getattribute__(name[5:])
                elif 'metrics' in name:
                    report_dict[name] = self.metrics[name[8:]][-1]
                elif 'optimD' in name:
                    report_dict[name] = self.d_optim.param_groups[0][name[7:]]
                elif 'optimG' in name:
                    report_dict[name] = self.g_optim.param_groups[0][name[7:]]
                else:
                    report_dict[name] = self.__getattribute__(name)
            except:
                report_dict[name] = -99
        if not return_df:
            return report_dict
        else:
            return pd.DataFrame.from_dict(report_dict, orient='index').T

    @staticmethod
    def _get_list_of_metrics():
        metrics_list = ['total_iters', 'total_gen_iters',
                        'netG_loss', 'netD_loss',
                        'avg_D_real', 'avg_D_fake',
                        'rmse_avg', 'corr_avg', 'rmse_std', 'corr_std',
                        'rmse_num_avg', 'rmse_num_std', 'corr_num_avg', 'corr_num_std',
                        'rmse_cat_avg', 'rmse_cat_std', 'corr_cat_avg', 'corr_cat_std',
                        'rmse_pred_scatter', 'corr_pred_scatter',
                        'real_fake_acc', 'real_fake_auc', 'real_fake_f1',
                        'oversampling_acc', 'oversampling_auc', 'oversampling_f1',
                        'synthtraining_acc', 'synthtraining_auc', 'synthtraining_f1',
                        'real_test_pred_acc', 'real_test_pred_auc', 'real_test_pred_f1',
                        'netG_gradients', 'netD_gradients']
        return metrics_list


class WGANGP(BaseGAN):
    def __init__(self,
                 netG=None,
                 netD=None,
                 g_optim=None,
                 d_optim=None,
                 d_updates_per_g: int = 3,
                 gp_weight=10,
                 gp_with_embs: bool = False,
                 verbose: int = 1,
                 write_to_disk: bool = True,
                 print_every: int = 150,
                 compute_metrics_every: int = 150,
                 plot_every: int = 300,
                 save_model_every: int = 5000,
                 save_data_every: int = 10000,
                 prefix: str = None,
                 transformer=None,
                 num_cols=None,
                 cat_cols=None,
                 cat_dims=None):
        super(WGANGP, self).__init__(netG=netG, netD=netD,
                                     g_optim=g_optim, d_optim=d_optim,
                                     write_to_disk=write_to_disk,
                                     d_updates_per_g=d_updates_per_g,
                                     verbose=verbose, print_every=print_every,
                                     compute_metrics_every=compute_metrics_every,
                                     plot_every=plot_every,
                                     save_model_every=save_model_every, save_data_every=save_data_every,
                                     prefix=prefix,
                                     transformer=transformer, num_cols=num_cols, cat_cols=cat_cols, cat_dims=cat_dims)
        self.gp_weight = gp_weight
        self.gp_with_embs = gp_with_embs
        self.metrics['GP'] = []
        self.metrics['Distance'] = []

    def _netD_iter(self, X_batch, y_batch=None):
        self.netD.zero_grad()

        fake = self.netG.sample(self.batch_size, y=y_batch)

        d_real = self.netD(X_batch, y_batch).view(-1)
        d_fake = self.netD(fake, y_batch).view(-1)

        gradient_penalty = self._calc_gradient_penalty(X_batch, fake, y=y_batch)

        g_loss = d_fake.mean()
        distance = d_real.mean() - g_loss
        d_loss = -distance + (self.gp_weight * gradient_penalty)
        d_loss.backward()
        self.d_optim.step()

        self.metrics['total_iters'].append(self.total_iters)
        self.metrics['netD_loss'].append(-d_loss.data.numpy().item())
        self.metrics['avg_D_real'].append(d_real.data.mean().numpy().item())
        self.metrics['avg_D_fake'].append(d_fake.data.mean().numpy().item())
        self.metrics['GP'].append(gradient_penalty.data.numpy().item())
        self.metrics['Distance'].append(distance.data.numpy().item())

    def _netG_iter(self, X_batch, y_batch=None):

        self.g_optim.zero_grad()

        fake = self.netG.sample(self.batch_size, y=y_batch)
        d_fake = self.netD(fake, y_batch).view(-1)
        g_loss = - d_fake.mean()
        g_loss.backward()
        self.g_optim.step()

        self.total_gen_iters += 1
        for _ in range(self.d_updates_per_g):
            self.metrics['total_gen_iters'].append(self.total_gen_iters)
            self.metrics['netG_loss'].append(g_loss.data.item())

    def _calc_gradient_penalty(self, real_data, fake_data, y=None):

        epsilon = torch.rand(self.batch_size, 1)
        interpolated = epsilon * real_data.data + (1 - epsilon) * fake_data.data
        interpolated.requires_grad = True

        d_interpolated = self.netD(interpolated, y)

        gradients = torch_grad(outputs=d_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(d_interpolated.size()),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return ((gradients_norm - 1) ** 2).mean()

    def _print_metrics(self, n_iters, end='\n'):

        out = f"[{self.total_iters:5}/{n_iters}] LG:{self.metrics['netG_loss'][-1]:.3f} " \
              f"LD:{self.metrics['netD_loss'][-1]:.3f} " \
              f"D:{self.metrics['Distance'][-1]:.3f} " \
              f"GP:{self.metrics['GP'][-1]:.3f} "

        try:
            out += f"RMSEAVG:{self.metrics['rmse_avg'][-1]:.3f} " \
                   f"NUM:{self.metrics['rmse_num_avg'][-1]:.3f} " \
                   f"SynTraiAuc:{self.metrics['synthtraining_auc'][-1]:.3f} " \
                   f"RFAcc:{self.metrics['real_fake_acc'][-1]:.3f}  "
        except IndexError:
            pass

        print(out, end=end)

    def _plot_metrics(self, show=False):
        fig, axes = plt.subplots(3, 2)
        axes = axes.flatten()
        fig.set_size_inches((16, 9))
        legend_kwargs = {'fontsize': 'x-small', 'title_fontsize': 'small', 'loc': 'upper left'}

        axes[0].plot(self.metrics['netG_loss'][:], label='Gen')
        axes[0].plot(self.metrics['netD_loss'][:], label='Disc')
        axes[0].legend(title='Loss', **legend_kwargs)
        axes[0].set_xticks([])

        axes[2].plot(self.metrics['avg_D_real'], label='Real')
        axes[2].plot(self.metrics['avg_D_fake'], label='Fake')
        axes[2].legend(title='Mean(D)', **legend_kwargs)

        axes[1].plot(self.metrics['rmse_avg'], label='All columns')
        axes[1].plot(self.metrics['rmse_num_avg'], label='Num')
        axes[1].plot(self.metrics['rmse_cat_avg'], label='Cat')
        axes[1].plot(self.metrics['rmse_std'], label='All columns', linestyle=':')
        axes[1].plot(self.metrics['rmse_num_std'], label='Num', linestyle=':')
        axes[1].plot(self.metrics['rmse_cat_std'], label='Cat', linestyle=':')
        axes[1].plot(self.metrics['rmse_pred_scatter'], label='Pred')
        axes[1].legend(title='RMSE AVG/STD', **legend_kwargs)
        axes[1].set_xticks([])
        axes[1].set_ylim(top=0.2, bottom=0.0)

        axes[3].plot(self.metrics['corr_avg'], label='All columns')
        axes[3].plot(self.metrics['corr_num_avg'], label='Num')
        axes[3].plot(self.metrics['corr_cat_avg'], label='Cat')
        axes[3].plot(self.metrics['corr_std'], label='All columns', linestyle=':')
        axes[3].plot(self.metrics['corr_num_std'], label='Num', linestyle=':')
        axes[3].plot(self.metrics['corr_cat_std'], label='Cat', linestyle=':')
        axes[3].legend(title='CORR AVG/STD', **legend_kwargs)
        axes[3].set_ylim(bottom=0.8, top=1.01)

        axes[4].plot(self.metrics['Distance'], label='Distance')
        axes[4].legend(**legend_kwargs)

        axes[5].plot(self.metrics['GP'], label='Gradient Penalty')
        axes[5].legend(**legend_kwargs)

        plt.tight_layout()
        if show:
            plt.show()

    def get_report_row(self, return_df: bool = True):
        report_dict = super(WGANGP, self).get_report_row(return_df=False)
        report_dict['metrics.GP'] = self.metrics['GP'][-1]
        report_dict['metrics.distance'] = self.metrics['Distance'][-1]
        report_dict['gp_weight'] = self.gp_weight

        if not return_df:
            return report_dict
        else:
            return pd.DataFrame.from_dict(report_dict, orient='index').T


class Generator(nn.Module):
    def __init__(self,
                 output_dim: int,
                 hidden_layer_sizes: tuple = (100, 100,),
                 noise_dim: int = 30,
                 normal_noise: bool = False,
                 n_cross_layers: int = 0,
                 cat_output_dims: list = None,
                 condition: bool = False,
                 activation: str = 'relu',
                 dropout: float = 0,
                 layer_norm: bool = False,
                 condition_num_on_cat: bool = True,
                 use_external_embedding_layer: bool = False,
                 reduce_cat_dim: bool = True,
                 use_num_hidden_layer: bool = True,
                 num_activation: str = 'none',
                 cat_activation: str = 'gumbel-softmax',
                 gumbel_kwargs: dict = None,
                 hard_sampling: bool = True):
        super(Generator, self).__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.layer_norm = layer_norm
        self.n_cross_layers = n_cross_layers
        self.noise_dim = noise_dim
        self.normal_noise = normal_noise
        self.output_dim = output_dim
        self.cat_output_dims = cat_output_dims
        self.condition = condition
        self.training_iterations = 0
        self.cat_activation = cat_activation
        self.dropout = dropout
        self.hard_sampling = hard_sampling
        self.condition_num_on_cat = condition_num_on_cat
        self.use_external_embedding_layer = use_external_embedding_layer
        self.reduce_cat_dim = reduce_cat_dim
        self.use_num_hidden_layer = use_num_hidden_layer
        self.gumbel_kwargs = gumbel_kwargs if gumbel_kwargs is not None else {'tau': 0.66, 'hard': 'sampling_only'}

        self.activation_name = activation
        self.num_activation_name = num_activation
        self.activation = get_activation_by_name(activation)  # activation_dict[activation]
        self.num_activation = get_activation_by_name(num_activation)  # activation_dict[num_activation]

        self.activations_need_restoring = False

        self.hidden_layers = []

        input_to_final_dim = noise_dim + 1 if condition else noise_dim
        input_to_cross_layers = input_to_final_dim
        if len(hidden_layer_sizes) > 0:
            first_layer = nn.Linear(input_to_final_dim, hidden_layer_sizes[0])

            if self.layer_norm:
                self.hidden_layers = nn.ModuleList(
                    [first_layer] + [nn.Sequential(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]),
                                                   nn.LayerNorm(hidden_layer_sizes[i + 1])) for i in
                                     range(len(hidden_layer_sizes) - 1)])
            else:
                self.hidden_layers = nn.ModuleList(
                    [first_layer] + [nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]) for i in
                                     range(len(hidden_layer_sizes) - 1)])

            input_to_final_dim = hidden_layer_sizes[-1]

        if self.n_cross_layers > 0:
            self.cross_layers = nn.ModuleList([Cross(input_to_cross_layers) for _ in range(n_cross_layers)])

            input_to_final_dim += input_to_cross_layers

        self.cat_final_layers = None
        if cat_output_dims is not None:
            if cat_activation == 'softmax':
                self.cat_final_layers = nn.ModuleList(
                    [nn.Sequential(nn.Linear(input_to_final_dim, dim),
                                   nn.Softmax(dim=1)) for dim in cat_output_dims])
            elif cat_activation == 'gumbel_softmax':
                self.cat_final_layers = nn.ModuleList(
                    [nn.Sequential(nn.Linear(input_to_final_dim, dim),
                                   GumbelSoftmax(dim=1, **self.gumbel_kwargs)) for dim in cat_output_dims])
            else:
                raise ValueError(
                    f'Unknown cat_activation param {cat_activation}. Must be either"softmax" or "gumbel_softmax"')

        self.cat_reduction_layer = None
        self.emb_layers = None
        self.embedding_dims = None
        if condition_num_on_cat:
            if not self.reduce_cat_dim:
                input_to_final_dim += sum(cat_output_dims)
            else:

                self.embedding_dims = [int(min(np.ceil(cat_dim / 3), 20)) for cat_dim in cat_output_dims]
                self.emb_layers = nn.ModuleList([nn.Linear(cat_dim, emb_dim, bias=False)
                                                 for cat_dim, emb_dim in zip(cat_output_dims, self.embedding_dims)])
                self.cat_reduction_layer = nn.Linear(sum(self.embedding_dims), 16)
                input_to_final_dim += 16

        self.num_hidden_layer = None
        if self.use_num_hidden_layer:
            self.num_hidden_layer = nn.Linear(input_to_final_dim, 32)
            input_to_final_dim = 32

        self.final_layer = nn.Linear(input_to_final_dim, output_dim)

    def forward(self, x, y=None):
        if y is not None:
            y = (y + (y - 1))

        if self.activations_need_restoring:
            self._restore_activation_functions()

        if self.condition:
            x = torch.cat([x, y], dim=1)

        if self.n_cross_layers > 0:
            x0 = x

        for layer in self.hidden_layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)

            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=True)

        if self.n_cross_layers > 0:
            x_cross = x0
            for layer in self.cross_layers:
                x_cross = layer(x0, x_cross)
            x = torch.cat([x, x_cross], dim=1)

        if self.cat_final_layers is not None:
            x_cat = [layer(x) for layer in self.cat_final_layers]
            if self.hard_sampling and not self.training:
                x_cat = [torch.zeros_like(logits).scatter_(1, torch.multinomial(logits, 1), 1.0) for
                         logits in x_cat]
        else:
            x_cat = []

        if self.condition_num_on_cat:
            x_cat_cond = [_tensor.clone().detach() for _tensor in x_cat]
            if self.use_external_embedding_layer:
                pass
            if self.reduce_cat_dim:

                x_emb = []
                for cat_idx, layer in enumerate(self.emb_layers):
                    x_emb.append(layer(x_cat_cond[cat_idx]))

                print(x_emb)
                x_emb = self.cat_reduction_layer(torch.cat([*x_emb], dim=1))
                if self.activation is not None:
                    x_emb = self.activation(x_emb)
                x_cat_cond = [x_emb]

            x = torch.cat([x, *x_cat_cond], dim=1)

        if self.use_num_hidden_layer:
            x = self.num_hidden_layer(x)
            if self.activation is not None:
                x = self.activation(x)

        x_num = self.final_layer(x)
        if self.num_activation is not None:
            x_num = self.num_activation(x_num)
        x_out = torch.cat([x_num, *x_cat], dim=1)

        return x_out

    def sample(self, n=5000, y=None):

        if self.normal_noise:
            z = torch.randn(n, self.noise_dim)
        else:
            z = torch.rand(n, self.noise_dim)

        if self.condition:
            if isinstance(y, (np.ndarray, torch.Tensor)):
                if len(y) == n:
                    y = torch.Tensor(y)
                else:
                    raise ValueError(
                        f'y is not the right length. Expected y to be of length n but got len(y)="{len(y)}" and n="{n}".')
            elif y is None or y == '50-50':
                if y is None:
                    logging.warning(
                        'Generator is in conditional mode, yet no class has been supplied. Defaulting to 50/50 split.')
                n_ones = n // 2
                n_zeros = n - n_ones
                y = torch.cat([torch.zeros(n_zeros, 1), torch.ones(n_ones, 1)], dim=0)
            elif isinstance(y, (int, float)):
                if y == 1:
                    y = torch.ones(n, 1)
                elif y == 0:
                    y = torch.zeros(n, 1)
            else:
                raise ValueError(f'NetG is in conditional mode, yet unknown y value was passed. '
                                 f'Got "{y}"  of type "{type(y).__name__}", '
                                 f'expected one of: "1", "0", "50-50", or an array of length n.')
        else:
            if y is not None:
                logging.warning('y was specified even though generator is not in conditional mode.')
        return self.forward(z, y=y)

    def _remove_activation_functions(self):
        self.num_activation = None
        self.activation = None
        self.activations_need_restoring = True

    def _restore_activation_functions(self):
        self.activation = get_activation_by_name(self.activation_name)
        self.num_activation = get_activation_by_name(self.num_activation_name)
        if not self.activations_need_restoring:
            logging.warning('Generator activation functions got restored unnecessarily.')
        self.activations_need_restoring = False


class Discriminator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_layer_sizes: tuple = (100, 100,),
                 cat_input_dims: list = None,
                 embedding_dims: list = 'auto',
                 condition: bool = False,
                 activation: str = 'leaky_relu',
                 sigmoid_activation: bool = False,
                 layer_norm: bool = True,
                 n_cross_layers: int = 0,
                 noisy_num_cols: bool = False):
        super(Discriminator, self).__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.input_dim = input_dim
        self.cat_input_dims = cat_input_dims
        self.embedding_dims = embedding_dims
        self.condition = condition
        self.sigmoid_activation = sigmoid_activation
        self.layer_norm = layer_norm
        self.noisy_num_cols = noisy_num_cols
        self.n_cross_layers = n_cross_layers
        self.training_iterations = 0

        self.activation_name = activation
        self.activation = get_activation_by_name(activation)

        if self.embedding_dims is not None:
            if self.embedding_dims == 'auto':
                self.embedding_dims = [int(min(np.ceil(cat_dim / 3), 20)) for cat_dim in cat_input_dims]
            self.cat_input_dim = sum(self.embedding_dims)
            self.emb_layers = nn.ModuleList([nn.Linear(cat_dim, emb_dim, bias=False)
                                             for cat_dim, emb_dim in zip(cat_input_dims, self.embedding_dims)])

        elif self.cat_input_dims is not None:
            self.cat_input_dim = sum(self.cat_input_dims) if len(self.cat_input_dims) > 0 else 0
        else:
            self.cat_input_dim = 0

        self.hidden_layers = []

        input_to_final_dim = input_dim + 1 if condition else input_dim
        input_to_final_dim += self.cat_input_dim
        input_to_cross_layers = input_to_final_dim

        if len(hidden_layer_sizes) > 0:
            first_layer = nn.Linear(input_to_final_dim, hidden_layer_sizes[0])

            if layer_norm:
                self.hidden_layers = nn.ModuleList(
                    [first_layer] + [nn.Sequential(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]),
                                                   nn.LayerNorm(hidden_layer_sizes[i + 1])) for i in
                                     range(len(hidden_layer_sizes) - 1)])
            else:
                self.hidden_layers = nn.ModuleList(
                    [first_layer] + [nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]) for i in
                                     range(len(hidden_layer_sizes) - 1)])

            input_to_final_dim = hidden_layer_sizes[-1]

        if self.n_cross_layers > 0:
            self.cross_layers = nn.ModuleList([Cross(input_to_cross_layers) for _ in range(n_cross_layers)])

            input_to_final_dim += input_to_cross_layers

        self.final_layer = nn.Linear(input_to_final_dim, 1)

    def forward(self, x, y=None):
        if y is not None:
            y = (y + (y - 1))

        x_num = x[:, :self.input_dim]

        if self.noisy_num_cols:
            x_num = x_num + torch.empty_like(x_num).normal_(mean=0, std=0.01)

        if self.embedding_dims is not None:
            start_idx = self.input_dim
            x_emb = []
            for layer, cat_dim in zip(self.emb_layers, self.cat_input_dims):
                end_idx = start_idx + cat_dim
                x_emb.append(layer(x[:, start_idx:end_idx]))
                start_idx += cat_dim
            x = torch.cat([x_num, *x_emb], dim=1)
        elif self.cat_input_dims is not None:
            x_cat = x[:, self.input_dim:]
            x = torch.cat([x_num, x_cat], dim=1)
        else:
            x = x_num

        if self.condition:
            x = torch.cat([x, y], dim=1)

        if self.n_cross_layers > 0:
            x0 = x

        for layer in self.hidden_layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)

        if self.n_cross_layers > 0:
            x_cross = x0
            for layer in self.cross_layers:
                x_cross = layer(x0, x_cross)

            x = torch.cat([x, x_cross], dim=1)

        x = self.final_layer(x)
        if self.sigmoid_activation:
            x = torch.sigmoid(x)
        return x


class GumbelSoftmax(nn.Module):
    __constants__ = ['dim']

    def __init__(self, dim=None, tau=0.666, hard='sampling_only'):
        super(GumbelSoftmax, self).__init__()
        self.dim = dim
        self.tau = tau
        assert hard in ['sampling_only', 'always', 'never']
        self.hard = hard

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        if self.hard == 'sampling_only':
            return F.gumbel_softmax(input, tau=self.tau, hard=not self.training, dim=self.dim)
        elif self.hard == 'never':
            return F.gumbel_softmax(input, tau=self.tau, hard=False, dim=self.dim)
        else:
            return F.gumbel_softmax(input, tau=self.tau, hard=True, dim=self.dim)

    def extra_repr(self):
        return 'dim={dim}, tau={tau}, hard={hard}'.format(dim=self.dim, tau=self.tau, hard=self.hard)


class Cross(nn.Module):

    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features
        self.weights = nn.Parameter(torch.Tensor(input_features))

        nn.init.normal_(self.weights, mean=0, std=np.sqrt(2 / input_features))
        self.bias = nn.Parameter(torch.Tensor(input_features))
        nn.init.constant_(self.bias, 0.1)

    def forward(self, x0, x):
        x0xl = torch.bmm(x0.unsqueeze(-1), x.unsqueeze(-2))
        return torch.tensordot(x0xl, self.weights, [[-1], [0]]) + self.bias + x

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.input_features, self.input_features)


def get_activation_by_name(name: str):
    activation_dict = {'relu': F.relu,
                       'leaky_relu': F.leaky_relu,
                       'tanh': torch.tanh,
                       'sigmoid': torch.sigmoid,
                       'linear': F.linear,
                       'none': None}
    return activation_dict[name]
