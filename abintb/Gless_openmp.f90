   
   subroutine Gless(c_vec1,c_vec2,den,n_t,n_k,Gans,n_v)

    implicit none
    !f2py threadsafe
    include "omp_lib.h"

    integer*4, intent(in) :: n_t, n_k, n_v 
    complex*16, dimension(n_k,n_v,n_t), intent(in) :: c_vec1
    complex*16, dimension(n_k,n_v,n_t), intent(in) :: c_vec2
    real*8, dimension(n_k,2), intent(in) :: den 
    complex*16, dimension(n_t,n_t), intent(out) :: Gans 
    integer*4 :: t1 
    integer*4 :: t2 

    !$OMP PARALLEL DO shared(c_vec1,c_vec2,den,Gans)
    do t1 = 1, n_t
        do t2 = 1, n_t
            Gans(t1,t2) = sum(sum((c_vec1(:,:,t1))*c_vec2(:,:,t2),dim=2)*den(:,1)*den(:,2))
        end do
    end do
    !OMP END PARALLEL DO

   end subroutine Gless 

