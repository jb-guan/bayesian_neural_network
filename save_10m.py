import os
import time
import re, parse
import numpy as np
import scipy.constants
import beamdiag


class objective():
    def __init__(self,file_cfg='sample.cfg',path='D:\\guan\\scan_for_longacc1\\',para = [100,6.84,10,6,1,0,7,0],
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
        #product beamfile.bin .bit parameters.txt
        if self.para[4] == 0:
            bf_name = 'beamfile_gau.py' 
        else:
            bf_name = 'beamfile_tri.py'
        os.system('python {} {} {} {} {} {} {} {} {} {} {}'.format(bf_name,
            self.cfg_file,self.para[0],self.para[1],self.para[2],self.para[3],self.para[6],self.para[8],self.para[9],self.para[10],self.para[11]))
        
        self.updateFile(r'{}'.format(self.file_cfg), "time-limit = ", "time-limit = {}".format(self.phase+0.1))
        ###########################linear rise or decrease##############@#####################################################

        #save beamfile.bin;parameters.txt
        #self.file_save(file_list=['beamfile.bin','parameters.txt'],state='ini_')
        
        #run lcode
        os.system('mpiexec -n 95 ./lcode {}'.format(self.cfg_file))
        
        #################################################################################################################
        type_list = ['bit','bin','png','log']#,'dat','txt']
        for i in range(0,len(type_list)):
            os.system('rm -rf *.{}'.format(type_list[i]))

        #calculate parameters of electron beam
        tb_file = []
        files = os.listdir()
        for file in files:
            if file[0:2] == 'tb':
                tb_file.append(file)
        tb_file.sort(key=lambda x: int(x[2:7]))

        self.para.append(0)

        for file_name in tb_file:#range(10000,int(self.phase/10+1000),1000):
            i = int(file_name[2:7])
            op = self.calcul(phase = i)
            #fit_obj = op[0]*op[5]/(op[1]*op[2]*op[3]*op[4])

            energy = op[0]
            spread = op[1] * 1e-2
            z_rms = op[2]
            r_rms = op[3]
            emit = op[4]
            charge = op[5]
            brightness = energy/1000*charge/emit/spread

            self.para[-1] = i #i*10
            with open("recoder.txt", "a+") as f:
                f.write('para = {}\n'.format(self.para))
                f.write('op = {}\n'.format(op))
                f.write('fit_obj = {}\n'.format(brightness))
                f.write('\n')
        
        #delete output file
        type_list = ['swp']#,'txt']
        for i in range(0,len(type_list)):
            os.system('rm -rf *.{}'.format(type_list[i]))
        return brightness


if __name__ == '__main__':
    import random

    mode_shape = 0 #define bunch shape type, 0:gaussian; 1:triangular
    mode_plasma = 0 #define plasma density distribution, -1:negative; 0:uniform; 1:positive
    #density = 7e14 #plasma density, range(1e14,1e15) cm^-3
    gradient = 0 # for example:0.5 gradient means 1e14(init) to 1.5e14(final) over 10m diatance

    import scipy.constants
    import numpy as np
    c=scipy.constants.c
    e=scipy.constants.e
    me=scipy.constants.electron_mass
    mp=scipy.constants.proton_mass
    e0=scipy.constants.epsilon_0
    pi=np.pi

    for i in range(0,3000):
        Q = random.uniform(50,300)
        em = random.uniform(2,10)
        length = random.uniform(40,120)
        pos = random.uniform(5.8,6.8)
        spd = random.uniform(0.1,0.5)
        Ld = 40         #random.uniform(20,100)
        Rd = 200        #random.uniform(100,200)
        Qd = 2.34       #random.uniform(1,4)

        R_driver = Rd*1e-6
        density = 7         #(1/R_driver*c)**2*e0*me/e**2/1e6/1e14 #calculate densty;transfor to cm^-3/1e14 

        #a = [q,em,sig_z,posi]
        # n0 = density * 1e6 *1e14
        # op=(n0*e**2/e0/me)**.5 #omegap
        # k=op/c
        phase = 50000.1     #(int(80*k/10000)+1)*10000  #acc 80 meters
        value = [Q,em,length,pos,mode_shape,mode_plasma,density,gradient,spd,Rd,Qd,Ld,phase,i]
        runs = objective(path='/mnt/data/scan_for_longacc6/10m_save/',para=value, phase=phase)
        ts = time.time() 
        fit_obj = runs.obj_fun()
        te = time.time()-ts
        with open("time.txt", "a+") as f:
            f.write('{} {} {} {}\n'.format(i,value[-1],te,te/60))
            f.write('\n')
       

    

