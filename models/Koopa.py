import math
import torch
import torch.nn as nn
import copy


class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """
    def __init__(self, mask_spectrum):
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum
        
    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf*mask, dim=1)
        x_inv = x - x_var
        
        return x_var, x_inv
    

class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=128, 
                 hidden_layers=2, 
                 dropout=0.05,
                 activation='tanh'): 
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y
    

class KPLayer(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """
    def __init__(self): 
        super(KPLayer, self).__init__()
        
        self.K = None # B E E

    def one_step_forward(self, z, return_rec=False, return_K=False):
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E
        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)


        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred

        return z_pred
    
    def forward(self, z, pred_len=1):
        assert pred_len >= 1, 'prediction length should not be less than 1'
        z_rec, z_pred= self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds


class KPLayerApprox(nn.Module):
    """
    Find koopman transition of linear system by DMD with multistep K approximation
    """
    def __init__(self): 
        super(KPLayerApprox, self).__init__()
        
        self.K = None # B E E
        self.K_step = None # B E E

    def forward(self, z, pred_len=1):
        # z:       B L E, koopman invariance space representation
        # z_rec:   B L E, reconstructed representation
        # z_pred:  B S E, forecasting representation
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E

        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1) # B L E
        
        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            temp_z_pred, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]

        return z_rec, z_pred
    

class TimeVarKP(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """
    def __init__(self,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 seg_len=24,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None,
                 multistep=False,
                ):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder            
        self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.pred_len / self.seg_len)   # segment number of output
        self.padding_len = self.seg_len * self.freq - self.input_len
        # Approximate mulitstep K by KPLayerApprox when pred_len is large
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer() 

    def forward(self, x):
        # x: B L C
        B, L, C = x.shape

        res = torch.cat((x[:, L-self.padding_len:, :], x) ,dim=1)

        res = res.chunk(self.freq, dim=1)     # F x B P C, P means seg_len
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)   # B F PC

        res = self.encoder(res) # B F H
        x_rec, x_pred = self.dynamics(res, self.step) # B F H, B S H

        x_rec = self.decoder(x_rec) # B F PC
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C
        
        x_pred = self.decoder(x_pred)     # B S PC
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :] # B S C

        return x_rec, x_pred


class TimeInvKP(nn.Module):
    """
    Koopman Predictor with learnable Koopman operator
    Utilize lookback and forecast window snapshots to predict the future of time-invariant term
    """
    def __init__(self,
                 input_len=96,
                 pred_len=96,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None):
        super(TimeInvKP, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder
        self.decoder = decoder

        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init) # stable initialization
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())
    
    def forward(self, x):
        # x: B L C
        res = x.transpose(1, 2) # B C L
        res = self.encoder(res) # B C H
        res = self.K(res) # B C H
        res = self.decoder(res) # B C S
        res = res.transpose(1, 2) # B S C

        return res

class TimeInvMLP(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, period_len, mode):
        super(TimeInvMLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.period_len = period_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)


class Model(nn.Module):
    '''
    Koopman Forecasting Model
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        self.mask_spectrum = configs.mask_spectrum
        self.enc_in = configs.enc_in
        self.input_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.num_blocks = configs.num_blocks
        self.dynamic_dim = configs.dynamic_dim
        self.hidden_dim = configs.hidden_dim
        self.hidden_layers = configs.hidden_layers
        self.multistep = configs.multistep

        self.disentanglement = FourierFilter(self.mask_spectrum)

        # shared encoder/decoder to make koopman embedding consistent
        self.time_inv_encoder = MLP(f_in=self.input_len, f_out=self.dynamic_dim, activation='relu',
                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.pred_len, activation='relu',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_kps = self.time_var_kps = nn.ModuleList([
                                TimeInvKP(input_len=self.input_len,
                                    pred_len=self.pred_len, 
                                    dynamic_dim=self.dynamic_dim,
                                    encoder=self.time_inv_encoder, 
                                    decoder=self.time_inv_decoder)
                                for _ in range(self.num_blocks)])

        # shared encoder/decoder to make koopman embedding consistent
        self.time_var_encoder = MLP(f_in=self.seg_len*self.enc_in, f_out=self.dynamic_dim, activation='tanh',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len*self.enc_in, activation='tanh',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_kps = nn.ModuleList([
                    TimeVarKP(enc_in=configs.enc_in,
                        input_len=self.input_len,
                        pred_len=self.pred_len,
                        seg_len=self.seg_len,
                        dynamic_dim=self.dynamic_dim,
                        encoder=self.time_var_encoder,
                        decoder=self.time_var_decoder,
                        multistep=self.multistep)
                    for _ in range(self.num_blocks)])
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: B L C

        # Series Stationarization adopted from NSformer
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Koopman Forecasting
        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if forecast is None:
                forecast = (time_inv_output + time_var_output)
            else:
                forecast += (time_inv_output + time_var_output)

        # Series Stationarization adopted from NSformer
        res = forecast * std_enc + mean_enc

        return res
    
# class StatModel(nn.Module):
#     '''
#     Koopman Forecasting Model
#     '''
#     def __init__(self, configs):
#         super(StatModel, self).__init__()
#         self.mask_spectrum = configs.mask_spectrum
#         self.enc_in = configs.enc_in
#         self.period_len = configs.period_len
#         self.seq_len = int(configs.seq_len / self.period_len)
#         self.pred_len = int(configs.pred_len / self.period_len)
#         self.seg_len = int(self.seq_len / 2)
        
#         self.hidden_layers = configs.hidden_layers
#         self.dynamic_dim = configs.dynamic_dim
#         self.hidden_dim = configs.hidden_dim
#         self.multistep = configs.multistep
#         self.channels = configs.enc_in if configs.features == 'M' else 1
#         self.station_type = configs.station_type

#         self.epsilon = 1e-5

#         self.disentanglement = FourierFilter(self.mask_spectrum)

#         # shared encoder/decoder to make koopman embedding consistent
#         self.mean_inv_encoder = MLP(f_in=self.seq_len, f_out=self.dynamic_dim, activation='tanh',
#                     hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.mean_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.pred_len, activation='tanh',
#                            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.mean_inv_kps = TimeInvKP(input_len=self.seq_len,
#                                     pred_len=self.pred_len, 
#                                     dynamic_dim=self.dynamic_dim,
#                                     encoder=self.mean_inv_encoder, 
#                                     decoder=self.mean_inv_decoder)
        
#         self.std_inv_encoder = MLP(f_in=self.seq_len, f_out=self.dynamic_dim, activation='tanh',
#                     hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.std_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.pred_len, activation='tanh',
#                            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.std_inv_kps = TimeInvKP(input_len=self.seq_len,
#                                     pred_len=self.pred_len, 
#                                     dynamic_dim=self.dynamic_dim,
#                                     encoder=self.std_inv_encoder, 
#                                     decoder=self.std_inv_decoder)

#         # shared encoder/decoder to make koopman embedding consistent
#         self.mean_var_encoder = MLP(f_in=self.seg_len*self.enc_in, f_out=self.dynamic_dim, activation='relu',
#                            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.mean_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len*self.enc_in, activation='relu',
#                            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.mean_var_kps = TimeVarKP(enc_in=configs.enc_in,
#                         input_len=self.seq_len,
#                         pred_len=self.pred_len,
#                         seg_len=self.seg_len,
#                         dynamic_dim=self.dynamic_dim,
#                         encoder=self.mean_var_encoder,
#                         decoder=self.mean_var_decoder,
#                         multistep=self.multistep)
        
#         self.std_var_encoder = MLP(f_in=self.seg_len*self.enc_in, f_out=self.dynamic_dim, activation='relu',
#                            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.std_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len*self.enc_in, activation='relu',
#                            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.std_var_kps = TimeVarKP(enc_in=configs.enc_in,
#                         input_len=self.seq_len,
#                         pred_len=self.pred_len,
#                         seg_len=self.seg_len,
#                         dynamic_dim=self.dynamic_dim,
#                         encoder=self.std_var_encoder,
#                         decoder=self.std_var_decoder,
#                         multistep=self.multistep)
        
#         self.mean_output_decoder = MLP(f_in=2, f_out=1, activation='tanh',
#                     hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
#         self.std_output_decoder = MLP(f_in=2, f_out=1, activation='relu',
#                     hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
    
#     def normalize(self, input):
#         if self.station_type == 'adaptive':
#             # x_enc: B L C
#             bs, len, dim = input.shape
#             time_var_input, time_inv_input = self.disentanglement(input)
#             time_var_input = time_var_input.reshape(bs, -1, self.period_len, dim)
#             time_inv_input = time_inv_input.reshape(bs, -1, self.period_len, dim)
#             input = input.reshape(bs, -1, self.period_len, dim)
#             mean_inv = torch.mean(time_inv_input, dim=-2, keepdim=True)
#             std_inv = torch.std(time_inv_input, dim=-2, keepdim=True)
#             mean_var = torch.mean(time_var_input, dim=-2, keepdim=True)
#             std_var = torch.std(time_var_input, dim=-2, keepdim=True)
#             norm_input = (input - mean_inv - mean_var) / (std_inv + std_var + self.epsilon)
#             mean_inv = mean_inv.reshape(bs, -1, dim)
#             mean_var = mean_var.reshape(bs, -1, dim)
            
#             std_inv = std_inv.reshape(bs, -1, dim)
#             std_var = std_var.reshape(bs, -1, dim)
            
#             # Koopman Forecasting
#             mean_inv_output = self.mean_inv_kps(mean_inv)
#             _, mean_var_output = self.mean_var_kps(mean_var)
#             mean_concat = torch.cat([mean_inv_output, mean_var_output], dim=-1)
#             std_inv_output = self.mean_inv_kps(std_inv)
#             _, std_var_output = self.mean_var_kps(std_var)
#             std_concat = torch.cat([std_inv_output, std_var_output], dim=-1)
#             outputs_mean = self.mean_output_decoder(mean_concat)
#             outputs_std = self.std_output_decoder(std_concat)
#             # outputs_mean = mean_inv_output + mean_var_output
#             # outputs_std = std_inv_output + std_var_output
#             outputs = torch.cat([outputs_mean, outputs_std], dim=-1)
#             return norm_input.reshape(bs, len, dim), outputs
#         else:
#             return input, None

#     def de_normalize(self, input, station_pred):
#         if self.station_type == 'adaptive':
#             bs, len, dim = input.shape
#             input = input.reshape(bs, -1, self.period_len, dim)
#             mean = station_pred[:, :, :self.channels].unsqueeze(2)
#             std = station_pred[:, :, self.channels:].unsqueeze(2)
#             output = input * (std + self.epsilon) + mean
#             return output.reshape(bs, len, dim)
#         else:
#             return input

class StatModel(nn.Module):
    '''
    Koopman Forecasting Model
    '''
    def __init__(self, configs):
        super(StatModel, self).__init__()
        self.mean_mask_spectrum = configs.mean_mask_spectrum
        self.std_mask_spectrum = configs.std_mask_spectrum
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.seq_len = int(configs.seq_len / self.period_len)
        self.pred_len = int(configs.pred_len / self.period_len)
        self.seg_len = int(self.seq_len / 2)
        
        self.hidden_layers = configs.hidden_layers
        self.dynamic_dim = configs.dynamic_dim
        self.hidden_dim = configs.hidden_dim
        self.multistep = configs.multistep
        self.channels = configs.enc_in if configs.features == 'M' else 1
        self.station_type = configs.station_type

        self.epsilon = 1e-5

        self.mean_disentanglement = FourierFilter(self.mean_mask_spectrum)
        self.std_disentanglement = FourierFilter(self.std_mask_spectrum)

        self.mean_inv_MLP = TimeInvMLP(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in, period_len=self.period_len, mode='mean')
        self.std_inv_MLP = TimeInvMLP(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in, period_len=self.period_len, mode='std')
        self.weight = nn.Parameter(torch.ones(2, self.channels))

        # shared encoder/decoder to make koopman embedding consistent
        self.mean_var_encoder = MLP(f_in=self.seg_len*self.enc_in, f_out=self.dynamic_dim, activation='relu',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.mean_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len*self.enc_in, activation='relu',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.mean_var_kps = TimeVarKP(enc_in=configs.enc_in,
                        input_len=self.seq_len,
                        pred_len=self.pred_len,
                        seg_len=self.seg_len,
                        dynamic_dim=self.dynamic_dim,
                        encoder=self.mean_var_encoder,
                        decoder=self.mean_var_decoder,
                        multistep=self.multistep)
        
        self.std_var_encoder = MLP(f_in=self.seg_len*self.enc_in, f_out=self.dynamic_dim, activation='relu',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.std_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len*self.enc_in, activation='relu',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.std_var_kps = TimeVarKP(enc_in=configs.enc_in,
                        input_len=self.seq_len,
                        pred_len=self.pred_len,
                        seg_len=self.seg_len,
                        dynamic_dim=self.dynamic_dim,
                        encoder=self.std_var_encoder,
                        decoder=self.std_var_decoder,
                        multistep=self.multistep)
    
    def normalize(self, input):
        if 'adaptive' in self.station_type:
            # x_enc: B L C
            bs, len, dim = input.shape
            if self.station_type == 'minmax_adaptive':
                # instance minmax
                max = torch.max(input, dim=1, keepdim=True)[0]
                min = torch.min(input, dim=1, keepdim=True)[0]
                minmax = max - min
                input = (input - min) / (minmax + self.epsilon)
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(input, dim=-2, keepdim=True)
            std = torch.std(input, dim=-2, keepdim=True)
            norm_input = (input - mean) / (std + self.epsilon)
            input = input.reshape(bs, len, dim)
            mean_all = torch.mean(input, dim=1, keepdim=True)

            # Disentangle time-variant and time-invariant components of mean and std
            mean_var, mean_inv = self.mean_disentanglement(mean.reshape(bs, -1, dim))
            std_var, std_inv = self.std_disentanglement(std.reshape(bs, -1, dim))

            mean_inv_output = self.mean_inv_MLP(mean_inv - mean_all, input - mean_all) * self.weight[0] + mean_all * \
                           self.weight[1]
            std_inv_output = self.std_inv_MLP(std_inv, input)
            
            # Time variant koopman
            _, mean_var_output = self.mean_var_kps(mean_var)
            _, std_var_output = self.std_var_kps(std_var)
            outputs_mean = mean_inv_output + mean_var_output
            outputs_std = std_inv_output + std_var_output
            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)
            if self.station_type == 'minmax_adaptive':
                min_max = torch.cat([min, (minmax + self.epsilon)], dim=-1)
                return norm_input.reshape(bs, len, dim), outputs, min_max
            else:
                return norm_input.reshape(bs, len, dim), outputs
            # return norm_input.reshape(bs, len, dim), outputs
        else:
            return input, None

    def de_normalize(self, input, station_pred, minmax_pred=None):
        if 'adaptive' in self.station_type:
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)
            std = station_pred[:, :, self.channels:].unsqueeze(2)
            output = input * (std + self.epsilon) + mean
            output = output.reshape(bs, len, dim)
            if minmax_pred is not None:
                min = minmax_pred[:,:,0].reshape(bs,-1,1)
                min_max = minmax_pred[:,:,1].reshape(bs,-1,1)
                output = output * min_max + min
            return output
        else:
            return input