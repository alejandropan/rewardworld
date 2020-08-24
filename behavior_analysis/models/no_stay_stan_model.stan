// finished script. TODO: debug!
// calculate posteriors outside

functions {
    vector multi_softmax(vector qrl,  real beta) {
        real p_norm;
        vector[2] p;
        p_norm = exp(qrl[2]/beta) ./ (exp(qrl[1]/beta) + exp(qrl[2]/beta));
        p[1] = p_norm;
        p[2] = 1 - p_norm;
        return p;
        }
        
        
    vector calculate_ql_qr(real[,] q, vector cp) {
        vector[2] qlr;
        qlr[1] = q[1,1] .* cp[1] + q[2,1] .* cp[2];
        qlr[2] = q[1,2] .* cp[1] + q[2,2] .* cp[2];
        return qlr;
        }
        
    
    real contrast_posterior(real x, real xc, real theta, real sc, int x_i){
            return normal_cdf(x, 0, theta) * normal_lpdf(x | sc, theta);
            }

}




data {
    int NT; // this gives the MAX len of trials per session
    int NS; // max number of sessions per animal
    int NA; // number of animals
    int NS_all[NA]; // a vector with number sessions per subject
    int NT_all[NA, NS]; // trial lengths per session per animal size NA x NS
    int r[NA,NS,NT]; // rewards of the animal
    int l[NA,NS,NT]; // lasers the animal received
    real sc[NA,NS,NT]; // signed contrast
    int c[NA,NS,NT]; // choices of the animal
}

transformed data {
    int x_i[0]; // For integration a x_i value needs to be passed even if unused. Recommended solution: generate a 0 length variable
}


parameters {

// Population level 
	// inverse temperature
    real betam; // mean and sd of overarching distribution, this one is not fit -  4 because you nead (mean, sd) for drawing separately a mean and a sd for the population level
    real<lower=0> betasd;
    real<lower=0> beta_sd_m; // mean of the subject level sd
    real<lower=0> beta_sd_sd; // sd of the subject level sd
        
	// learning rate 
    real alpham;
    real<lower=0> alphasd;
    real<lower=0> alpha_sd_m; // mean and sd of the subject level sd
    real<lower=0> alpha_sd_sd; // sd of the subject level sd
		
	// sensory noise 
    real<lower=0>  sensorym;
    real<lower=0> sensorysd;
    real<lower=0> sensory_sd_m; // mean and sd of the subject level sd
    real<lower=0> sensory_sd_sd; // sd of the subject level sd
        
    // laser parameter
    real laserm;
    real<lower=0> lasersd; 
    real<lower=0> laser_sd_m; // mean and sd of the subject level sd
    real<lower=0> laser_sd_sd; // sd of the subject level sd


// for each mouse
	// inverse temperature
	real beta_mice_m[NA];
	real<lower=0>beta_mice_sd[NA];
	
	// learning rate 	
	real alpha_mice_m[NA];
	real<lower=0>alpha_mice_sd[NA];
	
	// sensory noise 	
	real<lower=0> sensory_mice_m[NA];
	real<lower=0>sensory_mice_sd[NA];

	// laser parameter		
	real laser_mice_m[NA];
	real<lower=0>laser_mice_sd[NA];

// for each mouse's session
	real betas[NA,NS]; // matrix with one b per mouse x session combination
	real alphas[NA,NS]; // matrix with one b per mouse x session combination
	real<lower=0> sensories[NA, NS]; // matrix with one b per mouse x session combination
	real lasers[NA, NS]; // matrix with one b per mouse x session combination
}


model {
        betam ~ normal(0,2);
        alpham ~ normal(0,2);
        sensorym ~ lognormal(0,2); // sernsory need to be postive
        laserm ~ normal(0,2);

        betasd ~ normal(0,2);
        alphasd ~ normal(0,2);
        sensorysd ~ lognormal(0,2);
        lasersd ~ normal(0,2);
        
        beta_sd_m ~ normal(0,2);
        alpha_sd_m ~ normal(0,2);
        sensory_sd_m ~ lognormal(0,2);
        laser_sd_m ~ normal(0,2);


        beta_sd_sd ~ normal(0,2);
        alpha_sd_sd ~ normal(0,2);
        sensory_sd_sd ~ lognormal(0,2);
        laser_sd_sd ~ normal(0,2);

        for (a in 1:NA) {
			beta_mice_m[a] ~ normal(betam,betasd);
			beta_mice_sd[a] ~ normal(beta_sd_m, beta_sd_sd);
			
			alpha_mice_m[a] ~ normal(alpham,alphasd);
			alpha_mice_sd[a] ~ normal(alpha_sd_m, alpha_sd_sd); 
			
			sensory_mice_m[a] ~ lognormal(sensorym,sensorysd);
			sensory_mice_sd[a] ~ lognormal(sensory_sd_m, sensory_sd_sd); 
			
			laser_mice_m[a] ~ normal(laserm,lasersd);
			laser_mice_sd[a] ~ normal(laser_sd_m, laser_sd_sd);
			
            for (s in 1:NS_all[a]) {
            
            	real alpha;
            	real q[2,2]; // I use 4 q values                 
                
                betas[a][s] ~ normal(beta_mice_m[a], beta_mice_sd[a]);
                alphas[a][s] ~ normal(alpha_mice_m[a], alpha_mice_sd[a]);
                sensories[a][s] ~ lognormal(sensory_mice_m[a], sensory_mice_sd[a]);
                lasers[a][s] ~ normal(laser_mice_m[a], laser_mice_sd[a]);

                for (i in 1:2) {
                    for  (j in 1:2){
                        q[i,j] = 0;
                        } 
                        }

                for (t in 1:NT_all[a,s]) {
                        vector[2] cp;  // contrast posterior
                        vector[2] qlr; // ql and qr for every trial
                        vector[2] p; // probability of a right choice
                        real d; // delta with laser
                        real bs_right;
                        real pchoice;
                        
                        // solution until integrate_1d is upgraded
                        real estep;
                        for  (i  in 1:1000){
                            estep =  -1 + (0.002 * i);
                            bs_right = 0;
                            bs_right += normal_cdf(estep, 0, sensories[a][s]) * normal_lpdf(estep| sc[a,s,t], sensories[a][s]) * 0.002;
                            }
                        cp[1]=1 - bs_right;
                        cp[2]=bs_right;
                        //
                        qlr = calculate_ql_qr(q, cp);
                        p = multi_softmax(qlr, betas[a][s]); // probability of a right choice
                        //print("p after bernoulli",bernoulli_logit_lpmf( c[a,s,t] | p));
                        pchoice = p[c[a,s,t]+1];
                        d = (r[a,s,t] + (l[a,s,t] * lasers[a][s])) - qlr[c[a,s,t]+1]; // +1 due not 0 indexing
                        //print("delta",d)
                        for (b in 1:2) {
                            q[b, c[a,s,t]+1] +=   cp[b] * alphas[a][s] * d; // alpha taken out for rescale (added back to be //consistent with intercept_last	
                            }
                        //print("q",q)
                        }
            	}
        }
}