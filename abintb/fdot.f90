   
subroutine fdot(mtx1,mtx2,ans,l,m,n)

    implicit none
    integer, intent(in) :: l,m,n
    real, intent(in) :: mtx1(l,m)
    real, intent(in) :: mtx2(m,n)
    real, intent(out) :: ans(l,n)

    ans =  MATMUL(mtx1,mtx2)

end subroutine fdot 

