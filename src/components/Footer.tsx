import React from 'react';
import { Flower2, Facebook, Twitter, Instagram, Youtube } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gradient-to-r from-primary-red to-red-800 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Logo and Description */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <Flower2 className="h-8 w-8 text-primary-gold" />
              <h3 className="font-serif text-xl font-bold">Kolam Pattern Recognition</h3>
            </div>
            <p className="text-red-100 mb-4 font-sans">
              Preserving and celebrating the beautiful art of Kolam through modern technology. 
              Discover, recognize, and recreate traditional South Indian patterns with our 
              AI-powered system.
            </p>
            <div className="flex space-x-4">
              <a href="#" className="text-red-200 hover:text-primary-gold transition-colors">
                <Facebook className="h-5 w-5" />
              </a>
              <a href="#" className="text-red-200 hover:text-primary-gold transition-colors">
                <Twitter className="h-5 w-5" />
              </a>
              <a href="#" className="text-red-200 hover:text-primary-gold transition-colors">
                <Instagram className="h-5 w-5" />
              </a>
              <a href="#" className="text-red-200 hover:text-primary-gold transition-colors">
                <Youtube className="h-5 w-5" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-serif text-lg font-semibold mb-4 text-primary-gold">Quick Links</h4>
            <ul className="space-y-2 font-sans">
              <li><a href="/" className="text-red-100 hover:text-white transition-colors">Home</a></li>
              <li><a href="/upload" className="text-red-100 hover:text-white transition-colors">Upload Pattern</a></li>
              <li><a href="/gallery" className="text-red-100 hover:text-white transition-colors">Gallery</a></li>
              <li><a href="/about" className="text-red-100 hover:text-white transition-colors">About Kolam</a></li>
              <li><a href="/contact" className="text-red-100 hover:text-white transition-colors">Contact Us</a></li>
            </ul>
          </div>

          {/* Support */}
          <div>
            <h4 className="font-serif text-lg font-semibold mb-4 text-primary-gold">Support</h4>
            <ul className="space-y-2 font-sans">
              <li><a href="#" className="text-red-100 hover:text-white transition-colors">Help Center</a></li>
              <li><a href="#" className="text-red-100 hover:text-white transition-colors">Privacy Policy</a></li>
              <li><a href="#" className="text-red-100 hover:text-white transition-colors">Terms of Service</a></li>
              <li><a href="#" className="text-red-100 hover:text-white transition-colors">Accessibility</a></li>
            </ul>
          </div>
        </div>

        <div className="border-t border-red-600 mt-8 pt-8 text-center">
          <p className="text-red-100 font-sans">
            © 2025 Kolam Pattern Recognition & Recreation System. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
