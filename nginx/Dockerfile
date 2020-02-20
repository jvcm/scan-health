# Use the Nginx image
FROM nginx

# Remove the default nginx.conf
RUN rm /etc/nginx/conf.d/default.conf

# Replace with my own nginx.conf
COPY nginx.conf /etc/nginx/conf.d/
