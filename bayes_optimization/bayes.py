import os
import time
import re, parse
import numpy as np
import scipy.constants
from mllib.model import KerasSurrogate

class objective():
    def __init__(self,file_cfg='sample.cfg',path='/mnt/data/bayes_node6/',para = [100,6.84,10,6,1,0,7,0],
                species=-1, phase=50000):
        self.file_cfg = file_cfg
        self.path = path
        self.cfg_file = self.path + self.file_cfg
        self.para = para
        self.species = species
        self.phase = phase

    def find(self,cfg, par):
        ans=re.search('\s' + par + '\s?=\s?[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', cfg.decode('utf-8')) #.decode('utf-8') new
        return float(ans.group(0).replace(par, '').replace('=', ''))


    def find_para_cfg(self):
        config=open(self.cfg_file, 'rb').read() #r-->rb new
        self.r_size=self.find(config, 'window-width')
        self.xi_size=self.find(config, 'window-length')
        self.dr=self.find(config, 'r-step')
        self.dz=self.find(config, 'xi-step')
        self.t_step=self.find(config, 'time-step')
        self.t_max=self.find(config, 'time-limit')
        self.save_beam=self.find(config, 'saving-period')
        self.save_output=self.find(config, 'output-time-period')
        self.subto = self.find(config,'subwindow-xi-to')
        self.subfrom = self.find(config,'subwindow-xi-from')


        with open(self.path+'parameters.txt') as f:
            text = f.read()
            text = re.sub(r'#.*', "", text)  # remove comments
            text = re.sub(r'[ \t]', "", text)  # remove spaces and tabs
            self.n0=parse.search("n_plasma={:g}", text)[0]
        
        #calculate the units
        self.c=scipy.constants.c
        e=scipy.constants.e
        m=scipy.constants.electron_mass
        e0=scipy.constants.epsilon_0
        op=(self.n0*e**2/e0/m)**.5 #omegap
        self.t = 1/op #unit of time
        self.k=self.c/op #unit of length
        self.E0 = 0.511 #MeV
        self.unit_E = m*self.c*op/e #the unit of electrion field
        self.unit_Phi = m*self.c**2/e #unit of wakefield potential
        self.unit_energy =  self.n0*m*self.c**2 #unit of energy density

    def calcul(self,para=['z_rms','r_rms','emr','Q0'],phase=50000):
        self.find_para_cfg()
        output = []
        a = beamdiag.beamdiago(self.path,phase,self.species)

        if sum(a) == 0:
            return [1,10000,10000,10000,10000,1]
        pm = np.sqrt(a[4]**2)#+a[6]**2)
        delta_pm = np.sqrt(a[5]**2)

        energy = self.E0*np.sqrt(pm**2+1) #unit of E is MeV
        delta_E = delta_pm*pm/(pm**2+1) * 100 #energy spread
        #delta_E = delta_pm/pm*100
        output.append(energy)
        output.append(delta_E)
        ####################################################################
        list_para = ['z_rms','r_rms','emr','Q0']
        list_value_index = [2,3,10,16] #value index in the output of beamdiag
        self.find_para_cfg()

        index = 10000
        for value in para:
            for i in range(0,len(list_para)):
                if value == list_para[i]:
                    index = i
            if index == 10000:
                    print('Para:{} is not in the list_para'.format(value))
            
            output.append(a[list_value_index[index]])
        return output
        #para_list = ['energy','spread','z_rms','r_rms','emr','  Q0']
        #list_unit_name = ['MeV','%','um','um','mm.mrad','pC']

    def file_save(self,file_list,state='ini_'):
        #save beamfile.bin parameters.txt
        if not os.path.exists(self.path+'save_file'):
            os.system('mkdir -m 777 '+self.path+'save_file')

        file_name = ''
        for i in range(0,len(self.para)):
            file_name = file_name+'{:.2f}_'.format(self.para[i])

        for j in range(0,len(file_list)):
            os.system('cp '+self.path+file_list[j]+' '+self.path+'save_file/'+state+
                    file_name+file_list[j])

    #update file of .cfg , the plasma longitudinal shape part
    #replace all old_str to new_str in file 
    def updateFile(self,file,old_str,new_str):
        file_data = ""
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if old_str in line:
                    line = new_str+'\n'
                file_data += line
        with open(file,"w",encoding="utf-8") as f:
            f.write(file_data)
    
    def obj_fun(self):
        model_dir = 'G:\\reverseNN\\awake_run2\global_solution\\model0.997\\mode5.h5\\awake_forward_model'
        model_name = ''
        surr = KerasSurrogate.load(model_dir,model_name)
        [energy,spread,emit,charge] = surr.predict(self.value[0:5].append(10.0))
        
        return energy/1000*charge/spread/emit






if __name__ == '__main__':
    import scipy.constants
    import numpy as np
    c=scipy.constants.c
    e=scipy.constants.e
    me=scipy.constants.electron_mass
    mp=scipy.constants.proton_mass
    e0=scipy.constants.epsilon_0
    pi=np.pi

    import optuna
    import joblib
    from optuna.integration import SkoptSampler

    def object_trial(trial):
        Q = trial.suggest_uniform('Q', 50, 300)
        em = trial.suggest_uniform('em', 2, 6)
        length = trial.suggest_uniform('length', 30, 80)
        pos = trial.suggest_uniform('pos', 5.8, 6.8)
        spd = trial.suggest_uniform('spd', 0.1, 0.5)
        mode_shape = 0 #trial.suggest_int('shape', 0, 1) #define bunch shape type, 0:gaussian; 1:triangular
        mode_plasma = 0 #trial.suggest_int('plasma',-1,1) #define plasma density distribution, -1:negative; 0:uniform; 1:positive
        #density = trial.suggest_discrete_uniform('density', 1, 10,0.1)*1e14 #plasma density, range(1e14,1e15) cm^-3
        gradient = 0 #trial.suggest_uniform('grad',0,0.5) # for example:0.5 gradient means 1e14(init) to 1.5e14(final) over 10m diatance
        Ld = trial.suggest_uniform('Ld', 20, 100)
        Qd = trial.suggest_uniform('Qd', 1.00, 6.00)
        Rd = trial.suggest_uniform('Rd', 50, 200)
        R_driver = Rd*1e-6
        density = (1/R_driver*c)**2*e0*me/e**2/1e6/1e14 #calculate densty;transfor to cm^-3/1e14

        n0 = density * 1e6 *1e14
        op=(n0*e**2/e0/me)**.5 #omegap
        k=op/c
        phase = (int(80*k/10000)+1)*10000  #acc 80 meters
        value = [Q,em,length,pos,spd,mode_shape,mode_plasma,density,gradient,spd,Rd,Qd,Ld,phase,i]

        #value = [Q,em,length,pos,mode_shape,mode_plasma,density,gradient,Ld,Qd,Rd]
        runs = objective(path = '/mnt/data/bayes_node6/', para = value,phase=phase)
        fit_obj = runs.obj_fun()
        return fit_obj


    algo = 'TPE'

    if algo == "TPE":
        algo = optuna.samplers.TPESampler(n_startup_trials = 50, n_ei_candidates = 24)#默认最开始有10个观测值，每一次计算采集函数随机抽取24组参数组合
    elif algo == "GP":
        algo = SkoptSampler(skopt_kwargs={'base_estimator':'GP', #选择高斯过程
                                          'n_initial_points':20, #初始观测点10个
                                          'acq_func':'EI'} #选择的采集函数为EI，期望增量
                           )

    study_name = 'bayes_long6'

    i = 2
    study_multi_more = optuna.create_study(sampler = algo, #要使用的具体算法 sampler对样本进行抽样
                                directions=['maximize']
                                ,study_name=study_name, storage='sqlite:///bayes_long6.db', load_if_exists=True
                                )
    study_multi_more.optimize(object_trial, n_trials=2000, show_progress_bar=True)
    joblib.dump(study_multi_more, "/mnt/data/bayes_node6/bayes_long{}.pkl".format(i))
       

    

