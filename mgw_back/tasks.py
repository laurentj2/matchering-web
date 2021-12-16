"""
Matchering WEB - Handy Matchering 2.0 Containerized Web Application
Copyright (C) 2016-2021 Sergree

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from django.utils.text import get_valid_filename
from django_rq import job
import matchering as mg

from matchering_web.settings import MEDIA_ROOT
from mgw_back.models import MGSession
from mgw_back.utilities import get_directory, join, generate_filename

from pathvalidate import sanitize_filename
import soundfile as sf
import numpy as np
import importlib
import argparse
import warnings
import os.path
import librosa
import hashlib
import types
import shutil

import torch

import time
import glob
import cv2
import sys
import os

from lib.model_param_init import ModelParameters
from lib import vr as _inference
from lib import automation
from lib import spec_utils

def media(path):
    return join(MEDIA_ROOT, path)


class Paths:
    def __init__(self, target_path, target_title):
        self.title = get_valid_filename(target_title)
        self.folder = get_directory(target_path)
        self.result16 = join(self.folder, generate_filename("wav", 16, self.title))
        self.result24 = join(self.folder, generate_filename("wav", 24, self.title))
        self.preview_target = join(self.folder, generate_filename("flac"))
        self.preview_result = join(self.folder, generate_filename("flac"))


class SessionUpdater:
    def __init__(self, session: MGSession, paths: Paths):
        self.session = session
        self.paths = paths

    def __code(self, value):
        if len(value) >= 4:
            try:
                code = int(value[:4])
                if 2000 < code < 5000:
                    return code
            except ValueError:
                pass
        return 4201

    def info(self, value):
        code = self.__code(value)
        self.session.code = code
        if code == 2010:
            self.session.result16 = self.paths.result16
            self.session.result24 = self.paths.result24
            self.session.preview_target = self.paths.preview_target
            self.session.preview_result = self.paths.preview_result
        self.session.save()

    def warning(self, value):
        code = self.__code(value)
        self.session.warnings.create(code=code)


class inference:
    def __init__(self, _input, param, ptm, gpu=-1, hep='none', wsize=320, agr=0.07, tta=False, oi=False, de=False, v=False, spth='separated', fn='', pp=False, arch='default',
                pp_thres = 0.2, mrange = 32, fsize = 64):
        self.input = _input
        self.param = param
        self.ptm = ptm
        self.gpu = gpu
        self.hep = hep
        self.wsize = wsize
        self.agr = agr
        self.tta = tta
        self.oi = oi
        self.de = de
        self.v = v
        self.spth = spth
        self.fn = fn
        self.pp = pp
        self.arch = arch
        self.pp_thres = pp_thres
        self.mrange = mrange
        self.fsize = fsize
    def inference(self):
        nets = importlib.import_module('lib.nets' + f'_{self.arch}'.replace('_default', ''), package=None)
        # load model -------------------------------
        def loadModel():
            global mp, device, model
            try:
                print('loading model...', end=' ')
                mp = ModelParameters(self.param)
                device = torch.device('cpu')
                model = nets.CascadedASPPNet(mp.param['bins'] * 2)
                model.load_state_dict(torch.load(self.ptm, map_location=device))
                if torch.cuda.is_available() and self.gpu >= 0:
                    device = torch.device('cuda:{}'.format(self.gpu))
                    model.to(device)
            except Exception as e:
                return str(e)
            return True
        load_counter = 0
        while True:
            load_counter += 1
            if load_counter == 5:
                quit('An error has occurred: {}'.format(a))
            a = loadModel()
            if not type(a) == bool:
                print('Model loading failed, trying again...')
            else:
                del a
                break
        print('done')
        # stft of wave source -------------------------------
        print('stft of wave source...', end=' ')
        if self.fn != '':
            basename = self.fn
        else:
            basename = os.path.splitext(os.path.basename(self.input))[0]
        X_spec_m, input_high_end_h, input_high_end = spec_utils.loadWave(self.input, mp, hep=self.hep)
        print('done')
        vr = _inference.VocalRemover(model, device, self.wsize) # vr module
        if self.tta:
            pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': self.agr, 'split_bin': mp.param['band'][1]['crop_stop']})
        else:
            pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': self.agr, 'split_bin': mp.param['band'][1]['crop_stop']})
        if self.pp:
            print('post processing...', end=' ')
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv, thres=self.pp_thres, min_range=self.mrange, fade_size=self.fsize)
            print('done')
        # swap if v=True
        if self.v:
            stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
        else:
            stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        # deep ext stems!
        stems['di'] = 'DeepExtraction_Instruments'
        stems['dv'] = 'DeepExtraction_Vocals'
        y_spec_m = pred * X_phase # instruments
        v_spec_m = X_spec_m - y_spec_m # vocals

        #Instrumental wave upscale
        if self.hep == 'bypass':
            y_wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)
        elif self.hep.startswith('mirroring'):       
            input_high_end_ = spec_utils.mirroring(self.hep, y_spec_m, input_high_end, mp)
            y_wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end_)  
        else:
            y_wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
        
        v_wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        #saving files------------------------
        if self.de: # deep extraction
            model_name = os.path.splitext(os.path.basename(self.ptm))[0]
            print('inverse stft of {}...'.format(stems['inst']), end=' ')
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['inst'])), y_wave, mp.param['sr'])
            print('done')
            print('inverse stft of {}...'.format(stems['vocals']), end=' ')
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['vocals'])), v_wave, mp.param['sr'])
            print('done')

            print('Performing Deep Extraction...', end = ' ')
            if os.path.isdir('/content/tempde') == False:
                os.mkdir('/content/tempde')

            spec_utils.spec_effects(ModelParameters('modelparams/1band_sr44100_hl512.json'),
                                    [os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, 'Vocals')),
                                     os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, 'Instruments'))],
                                    '/content/tempde/difftemp_v',
                                    algorithm='min_mag')
            spec_utils.spec_effects(ModelParameters('modelparams/1band_sr44100_hl512.json'),
                                    ['/content/tempde/difftemp_v.wav',
                                     os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, 'Instruments'))],
                                    '/content/tempde/difftemp',
                                    algorithm='invert')
            os.rename('/content/tempde/difftemp.wav','/content/tempde/{}_{}_{}.wav'.format(basename, model_name, stems['di']))
            
            if os.path.isfile(self.spth+'/{}_{}_{}.wav'.format(basename, model_name, stems['di'])):
                os.remove(self.spth+'/{}_{}_{}.wav'.format(basename, model_name, stems['di']))
            shutil.move('/content/tempde/{}_{}_{}.wav'.format(basename, model_name, stems['di']),self.spth)
            # VOCALS REMNANTS
            
            if os.path.isfile(self.spth+'/{}_{}_{}.wav'.format(basename, model_name, stems['dv'])):
                os.remove(self.spth+'/{}_{}_{}.wav'.format(basename, model_name, stems['dv']))
            excess,_ = librosa.load('/content/tempde/difftemp_v.wav',mono=False,sr=44100)
            _vocal,_ = librosa.load(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, 'Vocals')),
                                    mono=False,sr=44100)
            # this isn't required, but just in case.
            excess, _vocal = spec_utils.align_wave_head_and_tail(excess,_vocal)
            sf.write(self.spth + '/{}_{}_{}.wav'.format(basename,model_name, stems['dv']),excess.T+_vocal.T,44100)
            print('Complete!')
        else: # args
            print('inverse stft of {}...'.format(stems['inst']), end=' ')
            model_name = os.path.splitext(os.path.basename(self.ptm))[0]
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['inst'])), y_wave, mp.param['sr'])
            print('done')
            print('inverse stft of {}...'.format(stems['vocals']), end=' ')
            print('done')
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['vocals'])), v_wave, mp.param['sr'])

@job
def process(session: MGSession):
    if session.code != 2002:
        return

    paths = Paths(session.target.file.name, session.target.title)
    updater = SessionUpdater(session, paths)

       try:
        process.inference(
            input=media(session.target.file.name),
            reference=media(session.target.file.name),
           
            results=[
                interference.y_wave(media(paths.result16)),
                interference.v_wave(media(paths.result24)),
            ],
            preview_target=interference.y_wave(media(paths.preview_target)),
            preview_result=interference.v_wave(media(paths.preview_result)),
        )
    except Exception as e:
        updater.info(str(e))
