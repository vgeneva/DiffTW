import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import tensorflow as tf
import pandas as pd
import math
from scipy.interpolate import interp1d
import tensorflow_probability as tfp
#from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
import re
import os

# in my big code, phi1 is train
# in my big code, phi0 is test

class DataLoader:
    def __init__(self, data_path_train, data_path_test):
        """
        Args:
            data_path_train: path to the training data CSV file
            data_path_test: path to the testing data CSV file
        """
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test
        print("DataLoader initialized with training path:", data_path_train)
        print("DataLoader initialized with testing path:", data_path_test)


    def extract_label(self, path):
        filename = os.path.basename(path)
        prefix = filename.split('_train')[0]

        # If digits exist, return them
        digits = re.findall(r'\d+', prefix)
        if digits:
            return digits[0]

        # Otherwise return letters
        letters = re.findall(r'[A-Za-z]+', prefix)
        return letters[0] if letters else None
        


    def load_data(self):
        # Load your data from the specified path
        """
        Loads the training and testing data from CSV files.
        Returns:
            data_train: DataFrame containing the training data
            data_test: DataFrame containing the testing data
        """
        self.data_train = pd.read_csv(self.data_path_train)
        self.data_test = pd.read_csv(self.data_path_test)
        print("Training data shape:", self.data_train.shape)
        print("Testing data shape:", self.data_test.shape)


        return self.data_train, self.data_test
    
    def preprocess_data(self):
        """
        Preprocesses the data by removing the first row and converting to numpy arrays.
        """
        # find the classification labels for train and test data
        # the labels are the headers of each column.
        # The labels will be the integer prior to the '.' in each column name.
        # For example, if a column is named '1.classA', the label is '1'.
        # Extract labels from column names
        # Get training labels 
        self.train_labels = [col.split('.')[0] for col in self.data_train.columns.tolist()]
        self.test_labels = [col.split('.')[0] for col in self.data_test.columns.tolist()]
        total_unique_train_labels = sorted(set(self.train_labels))
        total_unique_train_count = len(total_unique_train_labels)
        test_label = sorted(set(self.test_labels))

        print("Unique training labels:", total_unique_train_labels)
        print("Number of unique training labels:", total_unique_train_count)
        print(f"Number of train signals for each class: { {label: self.train_labels.count(label) for label in total_unique_train_labels} }")
        print("Extracted label for testing data:", self.extract_label(self.data_path_test))
        print("Number of test data signals:", len(test_label)) # don't think i need to see this?
        self.label_to_indices = {label: [i for i, l in enumerate(self.train_labels) if l == label] for label in total_unique_train_labels}


    def plot_sample(self,train_lbl, index=0):
        """Plots a sample from the training and testing data."""
        # get label for train selected:self.data_train.iloc[:, index]
        # Plot a sample from the training data
        #print("Extracted label for training data:", self.data_path_train[index, index])
        plt.figure(figsize=(10,8))
        plt.plot(self.data_train.iloc[:, index], color='black', label = f'Training Data {train_lbl}')
        plt.plot(self.data_test.iloc[:, index], color='blue', label = f'Testing Data {self.extract_label(self.data_path_test)}')  
        plt.title(f'Training Data and Testing Data')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

            

class proj_kernel:
    def __init__(self, 
                 num_features = 50,
                 bandwidth = .1):
        """
        Args
            num_features: number of random Fourier features
            bandwidth: bandwidth for the kernel
        """

        self.bandwidth = bandwidth
        self.num_features = num_features
        self.feature_coeff = 1/ np.sqrt(self.num_features)
        
        np.random.seed(3)
        self.fourier_sample = np.random.normal(scale =1, size=(self.num_features, 1))
        self.fourier_sample = tf.constant(self.fourier_sample, dtype = tf.float64)
        self.LU = None # will be set in make_proj_info()
        print("Projection kernel initialized.")

    def make_proj_info(self):
        
        """— Build G, compute A = I – Φ G⁻¹ Φᵀ, 
        spectral‐decompose, and store sqrt(Λ) Uᵀ. """
        
        # build 2x1 tensor [0 ,1 ]^T and compute G = \phi^T \phi, and G^-1:
        x_01 = np.array([0.,1.])
        x_01 = np.reshape(x_01, (len(x_01),1))
        x_01 = tf.constant(x_01, dtype = tf.float64)

        x_0 = np.array([0.])
        x_0 = np.reshape(x_0, (len(x_0),1))
        x_0 = tf.constant(x_0, dtype = tf.float64)
        
        x_1 = np.array([1.])
        x_1 = np.reshape(x_1, (len(x_1),1))  
        x_1 = tf.constant(x_1, dtype = tf.float64)
        self.x_1 = x_1
        self.x_0 = x_0
        self.x_01 = x_01

        G = self.exp_gaussian(x_01, x_01)
        
        G_inv = tf.linalg.inv(G)

        # compute (\phi(x_0) \phi(x_1)) in (p,2)     
        g0g1 = self._feature_map(x_01)
        
        # compute \phi(x_0) \phi(x_0)^T G \phi(x_0) \phi(x_1)
        gGg = tf.linalg.matmul(tf.linalg.matmul(g0g1, G_inv), tf.transpose(g0g1))
        #print(gGg.shape)
        I = tf.eye(2 * self.num_features, dtype = tf.float64)
        #print(I.shape)
        A = tf.subtract(I, gGg)

        # compute the eigenvalues and eigenvectors of A
        eig, U = LA.eigh(A)

        condition = eig < 1e-14
        #remove = tf.boolean_mask(eig, condition)
        ind_to_remove = np.where(condition)
        new_eig = np.delete(eig, ind_to_remove)
        new_U = np.delete(U, ind_to_remove, axis = 1)
        new_L = tf.linalg.diag(tf.sqrt(new_eig))
        LU = tf.linalg.matmul(new_L, tf.transpose(new_U))
        self.LU = LU
        self.LU = tf.cast(self.LU, dtype = tf.float64)

    def feature_map_0(self, x):
        """
        Args
            x: input data
            uses self._feature_map to compute the feature map
            then applies the projection matrix LU to the feature map
        Returns
            feature_map_0: transformed data
            vector of shape (2*num_features-2, n)
            for each x_i, we have 2*num_features-2 features
        """
        if self.LU is None:
            raise ValueError("Projection matrix not computed. Call make_proj_info() first.")
        p = self._feature_map(x)
        return tf.linalg.matmul(self.LU, p)
    

    def _feature_map(self, x):
        """
        Args
            x: input data (n, 1)
         
        calculates: 
            cos(ωx/bandwidth) and sin(ωx/bandwidth) where ω is the random Fourier sample
            then concatenates the two to get a feature map of shape (2*num_features, n)    
        Returns
            feature_map: transformed data
            vector of shape (2*num_features, n)
            for each x_i, we have 2*num_features features
        """
        x = tf.transpose(x)
        fourierx = tf.linalg.matmul(self.fourier_sample, x)
        cos_fourierx = self.feature_coeff*tf.cos(fourierx/self.bandwidth)
        sin_fourierx = self.feature_coeff*tf.sin(fourierx/self.bandwidth)
        feature_map = tf.concat([cos_fourierx, sin_fourierx], axis=0)
        return feature_map

    # make the exp_guassian kernel
    def exp_gaussian(self, 
                     x, y):
        """
        Args
            x: input data (n, 1)
            y: input data (n, 1)
        Returns
            kernel: the exponential Gaussian kernel between x and y"""

        return tf.linalg.matmul(tf.transpose(self._feature_map(x)), 
                                                     self._feature_map(y))
    
    def make_func(self, x, b):
        """
        Args
            x: input data (n-2, 1)
            b: input data (n-2, 1)
        Returns
            func: the learned function using the projection kernel and beta
            dimension (n-2, 1)
        """
        return tf.linalg.matmul(tf.transpose(b), self.feature_map_0(x))



 
class get_beta:
    def __init__(self, 
                 datapath_train=None,
                 datapath_test=None,
                 DataLoader_class=None,
                 proj_kernel_class=None,
                 indexphi0 = 60,
                 indexphi1 = 1,
                 phi0_array=None,
                 phi1_array=None):
        """
        Initialize the get_beta class.
        Args:
            datapath_train: path to the training data CSV file
            datapath_test: path to the testing data CSV file
            DataLoader_class: the DataLoader class for loading and preprocessing data
            proj_kernel_class: the projection kernel class for computing the projection matrix and feature map
            indexphi0: the column index for phi0 in the data, this is the test signal we want to align
            indexphi1: the column index for phi1 in the data, this is the training signal we want to align to
            phi0_array: optional preloaded phi0 array (if not provided, it will be loaded from the data)
            phi1_array: optional preloaded phi1 array (if not provided, it will be loaded from the data)
        Returns:
            Initializes the get_beta class with the loaded data, computed projection matrix, and initialized beta.
        """
        # ----------------------------
        if phi0_array is not None and phi1_array is not None:
            print("loading data directly from provided arrays.")
            self.phi0_test = np.array(phi0_array).flatten()
            self.phi1_train = np.array(phi1_array).flatten()
            print("Shapes of phi0_test and phi1_train:", self.phi0_test.shape, self.phi1_train.shape)
        

        # ---------------------------
        elif datapath_train and datapath_test and DataLoader_class:
            print("Loading data from CSV paths...")
            data_loader = DataLoader_class(datapath_train, datapath_test)
            data_train, data_test = data_loader.load_data()

            self.indexphi0 = indexphi0
            self.indexphi1 = indexphi1
            self.phi0_test = data_test.iloc[:, self.indexphi0].values
            self.phi1_train = data_train.iloc[:, self.indexphi1].values
            print("Shapes of phi0_test and phi1_train:", self.phi0_test.shape, self.phi1_train.shape)
            #label of train
            train_list = [col.split('.')[0] for col in data_train.columns.tolist()]
            self.train_label = train_list[self.indexphi1]
            first_label = train_list[indexphi1] # get the label of the first training data
            if '.' in first_label:
                extracted_label = first_label.split('.')[0]
            else:
                extracted_label = first_label
            print("Extracted label for first training data:", extracted_label)
            data_loader.plot_sample(train_lbl = extracted_label, index = indexphi1)
            data_loader.preprocess_data()

        else:
            raise ValueError("Please provide either data paths and DataLoader class, or preloaded phi0 and phi1 arrays.")
    

        # ---------------------------
        # Projection Kernel
        # ---------------------------
        self.proj_kernel = proj_kernel()
        self.make_proj_info = self.proj_kernel.make_proj_info() # compute the projection matrix, LU for make_func and for beta
        self.make_func = self.proj_kernel.make_func # function to be learned via beta
 


    def linear_inter(self, x, xp, fp):
        """
        Use TensorFlow Probability's interpolation for 1D linear interpolation.
        Args:
            x: Tensor of query points.
            xp: List/Tensor of known x-coordinates (must be equally spaced).
            fp: List/Tensor of known y-coordinates.
        Returns:
            Tensor (n, 1) Interpolated values at query points `x`.
        """
        # Ensure xp is a regular grid
        x = tf.cast(x, dtype=tf.float64)
        xp = tf.convert_to_tensor(tf.squeeze(xp), dtype=tf.float64) #need 0-d
        fp = tf.convert_to_tensor(tf.squeeze(fp), dtype=tf.float64) #need 0-d

        # Extract scalar min/max
        x_min = tf.reshape(xp[0], ())   # <-- force scalar
        x_max = tf.reshape(xp[-1], ()) 
        #x_min = tf.constant(xp[0], dtype=tf.float64)  # scalar
        print(f'x_min: {x_min}')
        #x_max = tf.constant(xp[-1], dtype=tf.float64) # scalar
        print(f'x_max: {x_max}')

        # Clip query points
        x = tf.clip_by_value(x, x_min, x_max)

        # Use tfp interp_regular_1d_grid
        interpolated_values = tfp.math.interp_regular_1d_grid(
            x=x,
            x_ref_min=x_min,
            x_ref_max=x_max,
            y_ref=fp,
            axis=0
        )
        return interpolated_values
    
    def i_traj(self, n, a= 0, b=1):
        """
        Args:
            n: number of points in the trajectory
            a: start of the interval
            b: end of the interval
        Returns:
            x: the initial trajectory, a column vector of shape (n, 1)
        """
        
        x_ini = tf.linspace(start = a, stop = b, num = n)
        #print(f'x_ini.shape: {x_ini.shape}')
        x = tf.reshape(x_ini, (n,1)) # reshape to column vector
        #print(f'x.shape: {x.shape}')
        return tf.cast(x, dtype = tf.float64)

    def forward_euler(self, BETA, phi0, num_steps): 
        """
        Args:
            phi0: the test signal (n, 1)
            num_steps: number of steps for the forward Euler method
        Returns:
            x: the estimated trajectory after num_steps iterations
            X: list of trajectories at each step (for visualization)
        """
        X = []
        x0 = self.i_traj(len(phi0))
        X.append(x0)
        #print(f'phi0 length: {len(phi0)}')
        x = self.i_traj(len(phi0))
        #print(f'initial trajectories.shape: {x.shape}')
        #alpha_temp = self.make_func(x, BETA)
        for i in range(1,num_steps):
            x += (1/(num_steps - 1)) * tf.transpose(self.make_func(x, BETA))
            x = tf.cast(x, dtype = tf.float64)
            x = tf.clip_by_value(x, 0.0, 1.0)  
            X.append(x)
        return x, X
    
    def phi_1_est(self, BETA, num_steps):
        """
        Estimate phi1 using the x1_est found using forward Euler method.
        phi0 is the length we need for forward euler, 
        but it would be the same using phi1 length
        Args:
        BETA: the beta vector used to compute the function
        Returns:
        phi1_est: the estimated phi1 values at the estimated x1 values
        """
        # Interpolate phi1 at the estimated x1 values
        #x1 = self.i_traj(len(self.phi1_train))
        x1 = tf.squeeze(self.i_traj(len(self.phi1_train)))
        #print(x1.shape)
        fe, _ = self.forward_euler(BETA, self.phi0_test, num_steps)
        #print(fe.shape)
        # Clip so interpolation remains inside domain
        #x_min = x1[0]
        #x_max = x1[-1]
            # Clip to domain using scalar values
        x_min = tf.cast(x1[0], dtype=tf.float64)   # scalar
        x_max = tf.cast(x1[-1], dtype=tf.float64)  # scalar
        fe_clipped = tf.clip_by_value(fe, x_min, x_max)

        # Interpolate
        return tfp.math.interp_regular_1d_grid(
            x=fe_clipped,
            x_ref_min=x_min,
            x_ref_max=x_max,
            y_ref=tf.cast(self.phi1_train, dtype=tf.float64),
         axis=0
        )
        #fe_clipped = tf.clip_by_value(fe, x_min, x_max)
        
        #return self.linear_inter(fe_clipped, x1, self.phi1_train) # interpolate phi1 at the estimated x1 values
    

    # can i put this in gradient descent?
    @tf.function
    def J_obj(self, BETA, num_steps, LAM):
        # -> estimated x @ time = 1
        # -> forward euler with self.beta
        # -> phi_1_est
        phi1_est = self.phi_1_est(BETA, num_steps)
        #print(f'phi1_est.shape: {phi1_est.shape}')
        #print(f'self.phi0_test shape: {self.phi0_transformed.shape}')
        phi0_t = tf.reshape(self.phi0_test, (-1, 1))
        #print(f'self.phi0_transformed shape: {phi0_t.shape}')
        phi_diff = tf.math.squared_difference(phi1_est, phi0_t)
        #print(f'phi_diff shape: {phi_diff.shape}')
        sum = tf.reduce_sum(phi_diff, axis = 0) # check axis
        #print(f'sum shape{ sum.shape}')
        n = (len(self.phi0_test))
        #print(f'n should be a number {n}')
        obj = 1/n * sum + (LAM * tf.transpose(BETA) @ BETA)
        #print(obj.shape)
        return obj

class Optimization:
    def __init__(self, get_beta):
        self.get_beta = get_beta
        #self.beta = tf.Variable(get_beta.beta, dtype=tf.float64)
        #self.optimizer = tf.optimizers.Adam(learning_rate=1/(2**9))
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1/(2**9))
    
    def optimize(self, beta_init=None, num_steps = None, num_iterations=10, LAM = 1e-4):

        beta_var = tf.Variable(beta_init, dtype=tf.float64)

        if num_steps is None:
            num_steps = 20

        print(f'num_steps: {num_steps}')
        self.num_steps = num_steps
        loss_list = []
        for i in range(num_iterations):
            with tf.GradientTape() as tape:
                loss = self.get_beta.J_obj(beta_var, self.num_steps, LAM)
            gradients = tape.gradient(loss, [beta_var])
            self.optimizer.apply_gradients(zip(gradients, [beta_var]))
            if i % 1 == 0:
                tf.print("Iteration", i, "Loss:", loss)
            loss_list.append(float(loss))

        return beta_var, loss_list


