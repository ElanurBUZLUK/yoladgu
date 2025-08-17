import { Injectable } from '@angular/core';
import { HttpEvent, HttpHandler, HttpInterceptor, HttpRequest } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ApiConfig, environment } from '../config/api.config';

@Injectable()
export class ApiBaseUrlInterceptor implements HttpInterceptor {
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    // Only modify requests that start with /api or are relative URLs
    if (req.url.startsWith('/api') || !req.url.startsWith('http')) {
      let apiUrl = req.url;
      
      // If it's a relative URL starting with /api, it's already correct for proxy
      if (req.url.startsWith('/api')) {
        // In production, we need to prepend the backend URL
        if (environment.production) {
          apiUrl = `${environment.backendUrl}${req.url}`;
        }
        // In development, proxy handles it, so keep as is
      } else {
        // If it's a relative URL not starting with /api, prepend the API path
        apiUrl = ApiConfig.getApiUrl(req.url);
      }

      const modifiedReq = req.clone({
        url: apiUrl
      });

      console.log(`API Request: ${req.url} -> ${modifiedReq.url}`);
      return next.handle(modifiedReq);
    }

    return next.handle(req);
  }
}