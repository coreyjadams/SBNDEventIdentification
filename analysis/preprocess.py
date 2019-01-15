import numpy
import os
import glob

from ROOT import larcv, std

class Preprocessor(object):

    def __init__(self):
        super(Preprocessor, self).__init__()

    def set_file(self, _file_name, _cfg=None):

        # Open this file with larcv:
        if _cfg is None:

            _cfg_str = '''
                  Verbosity:    2
                  EnableFilter: false
                  RandomAccess: false
                  ProcessType:  []
                  ProcessName:  []
                  AnaFile:      ""
                  IOManager: {
                    Verbosity:   2
                    Name:        "IOManager"
                    IOMode:      0
                    OutFileName: ""
                    InputFiles:  []
                    InputDirs:   []
                    StoreOnlyType: []
                    StoreOnlyName: []
                  }
                  ProcessList: {}
            '''
            _real_cfg = ''
            for line in _cfg_str.split('\n'):
                _real_cfg += line.strip() + ' '
            _cfg = larcv.PSet('ProcessDriver', _real_cfg)



        processor = larcv.ProcessDriver('ProcessDriver')
        processor.configure(_cfg)
        flist = std.vector('string')()
        flist.push_back(_file_name)

        processor.override_input_file(flist)

        processor.initialize()
        processor.process_entry(0)
        self._current_entry = 0
        self._io_manager = processor.io()
        self._processor = processor

        # Define the output data:
        self._output_dtypes = {
            'entry'  : numpy.int64,
            'neut'   : numpy.int8,
            'chrpi'  : numpy.int8,
            'neutpi' : numpy.int8,
            'prot'   : numpy.int8,
            'energy' : numpy.float32,
            # 'e_dep'  : numpy.float32,
            'nue_score2d'   : numpy.float64,
            'numu_score2d'  : numpy.float64,
            'nc_score2d'    : numpy.float64,
            'prot0_score2d' : numpy.float64,
            'prot1_score2d' : numpy.float64,
            'prot2_score2d' : numpy.float64,
            'chpi0_score2d' : numpy.float64,
            'chpi1_score2d' : numpy.float64,
            'ntpi0_score2d' : numpy.float64,
            'ntpi1_score2d' : numpy.float64,
            'nue_score3d'   : numpy.float64,
            'numu_score3d'  : numpy.float64,
            'nc_score3d'    : numpy.float64,
            'prot0_score3d' : numpy.float64,
            'prot1_score3d' : numpy.float64,
            'prot2_score3d' : numpy.float64,
            'chpi0_score3d' : numpy.float64,
            'chpi1_score3d' : numpy.float64,
            'ntpi0_score3d' : numpy.float64,
            'ntpi1_score3d' : numpy.float64,
        }



    def go_to_entry(self, entry):
        self._processor.process_entry(entry)
        return

    def event_loop(self, max_events = None):

        if max_events is None: max_events = self._io_manager.get_n_entries()

        self._output_arr = numpy.zeros(max_events,
            dtype={'names':self._output_dtypes.keys(),
                   'formats':self._output_dtypes.values()})

        while self._current_entry < max_events:
            self.go_to_entry(self._current_entry)
            self.process()
            self._current_entry += 1

        return

    def process(self):
        # this function is implement to do the details of the processing.
        self._output_arr[self._current_entry]['entry'] = self._current_entry
        self.get_true_labels()
        self.get_predictions()
        pass


    def get_true_labels(self):

        # Go into the event and get the true labels:
        neutrino_label = self._io_manager.get_data("particle", "neutID")
        chrpion_label  = self._io_manager.get_data("particle", "cpiID")
        neutpion_label = self._io_manager.get_data("particle", "npiID")
        proton_label   = self._io_manager.get_data("particle", "protID")

        mctruth = self._io_manager.get_data("particle","sbndneutrino")
        neutrino = mctruth.as_vector().front()
        self._output_arr[self._current_entry]['energy'] = neutrino.energy_init()


        self._output_arr[self._current_entry]['neut']   =  neutrino_label.as_vector().front().pdg_code()
        self._output_arr[self._current_entry]['chrpi']  =  chrpion_label.as_vector().front().pdg_code()
        self._output_arr[self._current_entry]['neutpi'] =  neutpion_label.as_vector().front().pdg_code()
        self._output_arr[self._current_entry]['prot']   =  proton_label.as_vector().front().pdg_code()

    def get_predictions(self):


        resnetneutrino = self._io_manager.get_data('meta','resnetneutrino')
        resnetchrpion = self._io_manager.get_data('meta','resnetchrpion')
        resnetntrpion = self._io_manager.get_data('meta','resnetntrpion')
        resnetproton = self._io_manager.get_data('meta','resnetproton')
        resnet3dneutrino = self._io_manager.get_data('meta','resnet3dneutrino')
        resnet3dchrpion = self._io_manager.get_data('meta','resnet3dchrpion')
        resnet3dntrpion = self._io_manager.get_data('meta','resnet3dntrpion')
        resnet3dproton = self._io_manager.get_data('meta','resnet3dproton')



        self._output_arr[self._current_entry]['nue_score2d']= resnetneutrino.get_double('0')
        self._output_arr[self._current_entry]['numu_score2d']= resnetneutrino.get_double('1')
        self._output_arr[self._current_entry]['nc_score2d']= resnetneutrino.get_double('2')
        self._output_arr[self._current_entry]['nue_score3d']= resnet3dneutrino.get_double('0')
        self._output_arr[self._current_entry]['numu_score3d']= resnet3dneutrino.get_double('1')
        self._output_arr[self._current_entry]['nc_score3d']= resnet3dneutrino.get_double('2')

        self._output_arr[self._current_entry]['prot0_score2d']= resnetproton.get_double('0')
        self._output_arr[self._current_entry]['prot1_score2d']= resnetproton.get_double('1')
        self._output_arr[self._current_entry]['prot2_score2d']= resnetproton.get_double('2')
        self._output_arr[self._current_entry]['prot0_score3d']= resnet3dproton.get_double('0')
        self._output_arr[self._current_entry]['prot1_score3d']= resnet3dproton.get_double('1')
        self._output_arr[self._current_entry]['prot2_score3d']= resnet3dproton.get_double('2')

        self._output_arr[self._current_entry]['chpi0_score2d']= resnetchrpion.get_double('0')
        self._output_arr[self._current_entry]['chpi1_score2d']= resnetchrpion.get_double('1')
        self._output_arr[self._current_entry]['chpi0_score3d']= resnet3dchrpion.get_double('0')
        self._output_arr[self._current_entry]['chpi1_score3d']= resnet3dchrpion.get_double('1')

        self._output_arr[self._current_entry]['ntpi0_score2d']= resnetntrpion.get_double('0')
        self._output_arr[self._current_entry]['ntpi1_score2d']= resnetntrpion.get_double('1')
        self._output_arr[self._current_entry]['ntpi0_score3d']= resnet3dntrpion.get_double('0')
        self._output_arr[self._current_entry]['ntpi1_score3d']= resnet3dntrpion.get_double('1')


    def persist(self, _output_file_name):

        numpy.save(_output_file_name, self._output_arr)

if __name__ == '__main__':
    _files = glob.glob('/data/sbnd/wire_pixel/event_id/*.root')
    for _file in _files:
        proc = Preprocessor()
        proc.set_file(_file)
        proc.event_loop()
        _out_name = os.path.basename(_file)
        _out_name = os.path.splitext(_out_name)[0] + '.npy'
        print _out_name
        proc.persist(_out_name)