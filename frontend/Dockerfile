FROM node:20-alpine
WORKDIR /app
# placeholder: static HTML, no build step required
COPY . .
EXPOSE 8080
CMD ["sh","-c","python3 -m http.server 8080 2>/dev/null || busybox httpd -f -p 8080"]

