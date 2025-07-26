import { CanActivateFn, Router } from '@angular/router';

export const authGuard: CanActivateFn = (route, state) => {
  const token = localStorage.getItem('token');
  if (!token) {
    window.location.href = '/login'; // veya uygun bir giriş rotası
    return false;
  }
  return true;
};
