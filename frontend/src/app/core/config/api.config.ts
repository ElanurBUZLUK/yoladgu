import { environment } from '../../../environments/environment';

export class ApiConfig {
  private static baseUrl = environment.apiUrl;

  static getApiUrl(endpoint: string = ''): string {
    // Remove leading slash if present to avoid double slashes
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
    
    // In development, proxy handles the routing
    // In production, use full URL
    if (environment.production) {
      return `${this.baseUrl}/${cleanEndpoint}`;
    } else {
      return `/api/v1/${cleanEndpoint}`;
    }
  }

  static getFullUrl(endpoint: string = ''): string {
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
    return `${environment.backendUrl}/api/v1/${cleanEndpoint}`;
  }

  // For debugging
  static getConfig() {
    return {
      production: environment.production,
      apiUrl: environment.apiUrl,
      backendUrl: environment.backendUrl,
      currentBaseUrl: this.baseUrl
    };
  }
}

// Export environment for backward compatibility
export { environment };